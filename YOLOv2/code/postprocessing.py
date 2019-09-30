# -*- coding: utf-8 -*-

from __future__ import division

import os
import math
import datetime
import cv2 as cv
import numpy as np
import parameters as para
from groundtruth import ANCHOR
from onehotcode import onehotdecode


def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits))


def box_decode(predictions,imgname,threshold):
    grid_s = predictions.shape[0]
    boxes = []
    for sh in range(grid_s):
        for sw in range(grid_s):
            one_cell_pred = predictions[sh,sw,0::]
            for b in range(para.NUM_ANCHORS):
                box = one_cell_pred[(b*(5+para.NUM_CLASSESS)):(b+1)*(5+para.NUM_CLASSESS)]
                prob = softmax(box[-para.NUM_CLASSESS::])
                max_prob = max(prob)
                confidence = 1/(1+math.exp(-box[4]))
                
                if (confidence*max_prob) >= threshold:
                    bx = (1/(1+math.exp(-box[0])) + sw)/grid_s
                    by = (1/(1+math.exp(-box[1])) + sh)/grid_s
                    bw = ANCHOR.Width[b]*math.exp(box[2])
                    bh = ANCHOR.Height[b]*math.exp(box[3])
                    
                    xmin = max(0.0,(bx - bw/2.0))
                    ymin = max(0.0,(by - bh/2.0))
                    xmax = min(1.0,(bx + bw/2.0))
                    ymax = min(1.0,(by + bh/2.0))
                    pred_class = onehotdecode(prob)
                    
                    pred_box = {'box':[xmin,ymin,xmax,ymax,max_prob],'className':pred_class}
                    boxes.append(pred_box)
                    
    result = {'imageName':imgname,'boxes':boxes}
    return result


def calculateIoU(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection
    
    if union<=0.0:
        iou = 0.0
    else:
        iou = 1.0*intersection / union
    return iou


def nms(result,threshold):
    class_list =[]
    final_pred_boxes = []
    boxes = result['boxes']
    
    for b in range(len(boxes)):
        class_list.append(boxes[b]['className'])
    class_list = np.unique(class_list)
    
    for name in class_list:
        box_coord = []
        for b in range(len(boxes)):
            if name == boxes[b]['className']:
                box_coord.append(boxes[b]['box'])       
        box_coord = np.array(box_coord)
        
        while box_coord.shape[0] > 0:                
            idx = np.argmax(box_coord[:,-1])
            keep_box = box_coord[idx,:]
            pred_box = {'box':keep_box,'className':name}
            final_pred_boxes.append(pred_box)
            
            box_coord = np.delete(box_coord,[idx],axis=0)
            if box_coord.shape[0] == 0:break
            
            suppre = []
            xmin0 = keep_box[0]
            ymin0 = keep_box[1]
            xmax0 = keep_box[2]
            ymax0 = keep_box[3]
            
            for b in range(box_coord.shape[0]):
                xmin1 = box_coord[b,:][0]
                ymin1 = box_coord[b,:][1]
                xmax1 = box_coord[b,:][2]
                ymax1 = box_coord[b,:][3]
                
                iou = calculateIoU(xmin0,ymin0,xmax0,ymax0,
                                   xmin1,ymin1,xmax1,ymax1)
                if iou > threshold:
                    suppre.append(b)
            box_coord = np.delete(box_coord,suppre,axis=0)
    detections = {'imageName':result['imageName'],'boxes':final_pred_boxes}
    return detections


def save_instance(detections):
    image_name = detections['imageName'][0]+'.'+para.PIC_TYPE
    read_dir = os.path.join(para.PATH,'data','annotation','images',image_name)
    write_dir = os.path.join(para.PATH,'pic')
    
    im = cv.imread(read_dir).astype(np.float32)
    im_h = im.shape[0]
    im_w = im.shape[1]
    
    im = cv.resize(im,(para.INPUT_SIZE,para.INPUT_SIZE)).astype(np.float32)
    for b in range(len(detections['boxes'])):
        box = detections['boxes'][b]['box']
        name = detections['boxes'][b]['className']
        
        xmin = int(box[0]*para.INPUT_SIZE)
        ymin = int(box[1]*para.INPUT_SIZE)
        xmax = int(box[2]*para.INPUT_SIZE)
        ymax = int(box[3]*para.INPUT_SIZE)
        prob = min(round(box[4]*100),100.0)
        txt = name +':'+ str(prob) + '%'
        
        font = cv.FONT_HERSHEY_PLAIN
        im = cv.rectangle(im,(xmin,ymin),(xmax,ymax),(255, 0, 0),1)
        im = cv.putText(im,txt,(xmin,ymin),font,1,(255,0,0),1)
    
    im = cv.resize(im,(im_w,im_h)).astype(np.float32)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')
    dst = os.path.join(write_dir,current_time+image_name)
    cv.imwrite(dst,im)
