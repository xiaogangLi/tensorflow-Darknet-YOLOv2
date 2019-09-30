# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:54:39 2019

@author: LiXiaoGang
"""
from __future__ import division

import os
import math
import numpy as np
import pandas as pd
import parameters as para
from parse import parse_size
from parse import parse_object
from onehotcode import onehotencode


ANCHOR = pd.read_csv(os.path.join(para.PATH,'anchor','anchor.txt'))
def get_groundtruth(xml):
    size_dict = parse_size(xml)
    rw = 1.0*para.INPUT_SIZE/size_dict['width']
    rh = 1.0*para.INPUT_SIZE/size_dict['height']
    
    gt = np.zeros([para.S,para.S,para.OUYPUT_CHANNELS],dtype=np.float32)
    mask = np.zeros([para.S,para.S],dtype=np.float32)
    truth_array = np.zeros([4*para.MAX_NUM_GT],dtype=np.float32)
    
    object_list = parse_object(xml)
    j = 0
    for box in object_list:
        box_class = box['classes']
        xmin =  box['xmin']*rw
        ymin =  box['ymin']*rh
        xmax =  box['xmax']*rw
        ymax =  box['ymax']*rh
        
        x_center = xmin + (xmax-xmin)/2.0
        y_center = ymin + (ymax-ymin)/2.0
                          
        bx = 1.0*x_center/para.DF    # para.DF = grid cell size(in pixel)
        by = 1.0*y_center/para.DF
        
        x_cell = int(bx)
        y_cell = int(by)
        
        m = max((bx - x_cell),1e-10)
        n = max((by - y_cell),1e-10)
        tx = math.log(m/(1-m))
        ty = math.log(n/(1-n))
        
        bw = 1.0*(xmax-xmin)/(para.INPUT_SIZE)
        bh = 1.0*(ymax-ymin)/(para.INPUT_SIZE)
        
        ###
        # If no object exists in that cell, the confidence scores should
        # be zero. Otherwise we want the confidence score to equal the
        # intersection over union (IOU) between the predicted box and the ground truth.
        ###
        
        # If object exists in that cell, the confidence score is tentatively initialized as 1.0.
        iou = 1.0    # confidence score = iou = sigmoid(to)
        bboxes = []
        class_onehotcode = np.squeeze(onehotencode([box_class+'_*']))
        for i in range(para.NUM_ANCHORS):
            pw = ANCHOR.Width[i]
            ph = ANCHOR.Height[i]
            tw = math.log(bw/pw)
            th = math.log(bh/ph)
            bbox = np.hstack(([tx,ty,tw,th,iou],class_onehotcode))
            bboxes = np.hstack((bboxes,bbox))
        current_gt = np.array(bboxes,dtype=np.float32)[None]
        gt[y_cell,x_cell,:] = current_gt
        mask[y_cell,x_cell] = 1.0
        
        if j < para.MAX_NUM_GT:
            truth_array[4*j:4*(j+1)] = np.divide([xmin,ymin,xmax,ymax],para.INPUT_SIZE,dtype=np.float32)
            j = j + 1
    return {'groundtruth':gt,'mask':mask,'truth_array':truth_array}