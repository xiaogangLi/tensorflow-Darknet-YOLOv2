# -*- coding: utf-8 -*-

from __future__ import division

import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import parameters as para
from darknet19 import darknet
from groundtruth import ANCHOR
from readbatch import mini_batch
from postprocessing import nms,box_decode,save_instance


def net_placeholder(batch_size=None):
    Input = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size,
                            para.INPUT_SIZE,
                            para.INPUT_SIZE,
                            para.CHANNEL],name='Input')
    
    Label = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size,
                            para.S,
                            para.S,
                            para.OUYPUT_CHANNELS],name='Label')
    
    Mask = tf.placeholder(dtype=tf.float32,
                          shape=[batch_size,
                          para.S,
                          para.S],name='Mask')
    
    Truth = tf.placeholder(dtype=tf.float32,
                           shape=[batch_size,4*para.MAX_NUM_GT],name='Truth')
    
    isTraining = tf.placeholder(tf.bool,name='Batch_norm')
    return Input,Label,Mask,isTraining,Truth


def isobject(Label,logits,i,j,k):
    max_iou = 0.0
    best_iou = 0.0
    iou_list = []
    loss_collection = []
    bbox_pred_highest_iou = tf.zeros(shape=[5+para.NUM_CLASSESS],dtype=tf.float32)
    bbox_gt_highest_iou = tf.zeros(shape=[5+para.NUM_CLASSESS],dtype=tf.float32)
    
    for b in range(para.NUM_ANCHORS):
        # tx,ty,tw,th,iou
        # ground truth
        bbox_gt = Label[i,j,k,b*(5+para.NUM_CLASSESS):(b+1)*(5+para.NUM_CLASSESS)]
        bx = (1/(1+tf.exp(-bbox_gt[0])) + k)/para.S
        by = (1/(1+tf.exp(-bbox_gt[1])) + j)/para.S
        bw = ANCHOR.Width[b]*tf.exp(bbox_gt[2])
        bh = ANCHOR.Height[b]*tf.exp(bbox_gt[3])
        bbox_gt_xmin = bx - bw/2.0
        bbox_gt_ymin = by - bh/2.0
        bbox_gt_xmax = bx + bw/2.0
        bbox_gt_ymax = by + bh/2.0
        
        # iou between anchor and groundtruth 
        # predictions              
        bbox_pred = logits[i,j,k,b*(5+para.NUM_CLASSESS):(b+1)*(5+para.NUM_CLASSESS)]  
        w = tf.maximum(0.0, tf.minimum(bbox_gt_xmax, (bbox_gt_xmin+ANCHOR.Width[b])) - tf.maximum(bbox_gt_xmin, bbox_gt_xmin))
        h = tf.maximum(0.0, tf.minimum(bbox_gt_ymax, (bbox_gt_ymin+ANCHOR.Height[b])) - tf.maximum(bbox_gt_ymin, bbox_gt_ymin))
        intersection = w*h
        union = (bbox_gt_xmax-bbox_gt_xmin)*(bbox_gt_ymax-bbox_gt_ymin)+(ANCHOR.Width[b])*(ANCHOR.Height[b])-intersection
        iou = tf.reduce_max([0.0,intersection/union])
        iou_list.append(iou)
        
        bbox_pred_highest_iou = tf.cond(max_iou<=iou,lambda:bbox_pred,lambda:bbox_pred_highest_iou) 
        bbox_gt_highest_iou = tf.cond(max_iou<=iou,lambda:bbox_gt,lambda:bbox_gt_highest_iou)
        max_iou = tf.cond(max_iou<=iou,lambda:iou,lambda:max_iou)

    # Here always be ANCHOR.Width[2] and ANCHOR.Height[2],which is not best.
    bx = (1/(1+tf.exp(-bbox_pred_highest_iou[0])) + k)/para.S
    by = (1/(1+tf.exp(-bbox_pred_highest_iou[1])) + j)/para.S
    bw = ANCHOR.Width[2]*tf.exp(bbox_pred_highest_iou[2])
    bh = ANCHOR.Height[2]*tf.exp(bbox_pred_highest_iou[3])
    bbox_pred_xmin = bx - bw/2.0
    bbox_pred_ymin = by - bh/2.0
    bbox_pred_xmax = bx + bw/2.0
    bbox_pred_ymax = by + bh/2.0     
    
    # best iou 
    w = tf.maximum(0.0, tf.minimum(bbox_gt_xmax, bbox_pred_xmax) - tf.maximum(bbox_gt_xmin, bbox_pred_xmin))
    h = tf.maximum(0.0, tf.minimum(bbox_gt_ymax, bbox_pred_ymax) - tf.maximum(bbox_gt_ymin, bbox_pred_ymin))
    intersection = w*h
    union = (bbox_gt_xmax-bbox_gt_xmin)*(bbox_gt_ymax-bbox_gt_ymin)+(bbox_pred_xmax-bbox_pred_xmin)*(bbox_pred_ymax-bbox_pred_ymin)-intersection
    best_iou = tf.reduce_max([0.0,intersection/union])

    loss_collection.append(tf.reduce_sum(para.OBJECT_SCALE*tf.square(tf.subtract(best_iou,tf.nn.sigmoid(bbox_pred_highest_iou[4])))))
    loss_collection.append(tf.reduce_sum(para.COORD_SCALE*tf.square(tf.subtract(bbox_gt_highest_iou[0:4],bbox_pred_highest_iou[0:4]))))
    loss_collection.append(tf.reduce_sum(para.CLASS_SCALE*tf.square(tf.subtract(bbox_gt_highest_iou[-para.NUM_CLASSESS::],tf.nn.softmax(bbox_pred_highest_iou[-para.NUM_CLASSESS::])))))
    loss_collection = tf.reduce_sum(loss_collection)
    return loss_collection
    
    
def noobject(Label,logits,truth,i,j,k):
    # tx,ty,tw,th,iou
    loss_collection = []
    for b in range(para.NUM_ANCHORS):
        # predctions              
        bbox_pred = logits[i,j,k,b*(5+para.NUM_CLASSESS):(b+1)*(5+para.NUM_CLASSESS)]
        bbox_gt = Label[i,j,k,b*(5+para.NUM_CLASSESS):(b+1)*(5+para.NUM_CLASSESS)]
        bx = (1/(1+tf.exp(-bbox_pred[0])) + k)/para.S
        by = (1/(1+tf.exp(-bbox_pred[1])) + j)/para.S
        bw = ANCHOR.Width[b]*tf.exp(bbox_pred[2])
        bh = ANCHOR.Height[b]*tf.exp(bbox_pred[3])
        bbox_pred_xmin = bx - bw/2.0
        bbox_pred_ymin = by - bh/2.0
        bbox_pred_xmax = bx + bw/2.0
        bbox_pred_ymax = by + bh/2.0
        
        iou_list = []
        for g in range(para.MAX_NUM_GT):
            # groundtruth
            gt = truth[i,:][4*g:4*(g+1)]
            g_xmin = gt[0]
            g_ymin = gt[1]
            g_xmax = gt[2]
            g_ymax = gt[3]
            
            # iou 
            w = tf.maximum(0.0, tf.minimum(g_xmax, bbox_pred_xmax) - tf.maximum(g_xmin, bbox_pred_xmin))
            h = tf.maximum(0.0, tf.minimum(g_ymax, bbox_pred_ymax) - tf.maximum(g_ymin, bbox_pred_ymin))
            intersection = w*h
            union = (g_xmax-g_xmin)*(g_ymax-g_ymin)+(bbox_pred_xmax-bbox_pred_xmin)*(bbox_pred_ymax-bbox_pred_ymin)-intersection
            iou = tf.reduce_max([0.0,intersection/union])
            iou_list.append(iou)
            
        max_iou = tf.reduce_max(iou_list)
        c_pred = tf.nn.sigmoid(bbox_pred[4])
        c_gt = bbox_gt[4]    # c_gt always is 0.
        c_loss = tf.cond(max_iou<para.MAX_IOU,lambda:(para.NOOBJECT_SCALE*tf.square(tf.subtract(c_gt,c_pred))),lambda:tf.cast(0.0,tf.float32))
        loss_collection.append(c_loss)
    loss_collection = tf.reduce_sum(loss_collection)
    return loss_collection
    
    
def net_loss(Label,mask,logits,truth):
    loss = []
    for i in range(para.BATCH_SIZE):
        loss_collection = []
        for j in range(para.S):
            for k in range(para.S):
                choice = tf.equal(mask[i,j,k],tf.constant(1.0,dtype=tf.float32))
                loss_cell = tf.cond(choice,lambda:isobject(Label,logits,i,j,k),lambda:noobject(Label,logits,truth,i,j,k))
                loss_collection.append(loss_cell)
        loss.append(tf.reduce_sum(loss_collection))
    loss = tf.reduce_mean(loss,name='Loss')
    return loss


def training_net():
    image,label,mask,isTraining,truth = net_placeholder(batch_size=None)
    logits = darknet(image,isTraining)
    loss = net_loss(label,mask,logits,truth)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):        
        train_step = tf.train.AdamOptimizer(para.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=5)
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter(os.path.join(para.PATH,'model'), sess.graph)
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        # restore model 
        if para.RESTORE_MODEL:
            if not os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH):
                print('Model does not exist！')
                sys.exit()
            ckpt = tf.train.get_checkpoint_state(para.CHECKPOINT_MODEL_SAVE_PATH)
            model = ckpt.model_checkpoint_path.split('/')[-1]
            Saver.restore(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,model))
            print('Successfully restore model:',model)
        
        for i in range(para.TRAIN_STEPS):
            batch = mini_batch(i,para.BATCH_SIZE,'train')
            feed_dict = {image:batch['image'], label:batch['label'], mask:batch['mask'],truth:batch['truth'], isTraining:True}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            print('===>Step %d: loss = %g ' % (i,loss_))
            
            # evaluate and save checkpoint
            if i % 500 == 0:
                write_instance_dir = os.path.join(para.PATH,'pic')
                if not os.path.exists(write_instance_dir):os.mkdir(write_instance_dir)
                j = 0
                while True:
                    
                    batch = mini_batch(j,1,'val')
                    feed_dict = {image:batch['image'],isTraining:False}
                    pred_output = sess.run(logits,feed_dict=feed_dict)
                    pred_output = np.squeeze(pred_output)
                    pred_output = box_decode(pred_output,batch['image_name'],para.CONFIDENCE_THRESHOLD)
                    pred_output = nms(pred_output,para.NMS_THRESHOLD)
                    
                    if j < min(10,batch['image_num']):save_instance(pred_output)
                    if j == batch['image_num']-1:break
                    j += 1
                
                if os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                    shutil.rmtree(para.CHECKPOINT_MODEL_SAVE_PATH)
                Saver.save(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,para.MODEL_NAME))             
            

def main():
    training_net()
     
if __name__ == '__main__':
    main()
