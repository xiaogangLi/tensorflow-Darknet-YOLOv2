# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:47:42 2019
@author: LiXiaoGang
"""
from __future__ import division

import os
import pandas as pd


PATH = os.path.dirname(os.getcwd())
LABELS = pd.read_csv(os.path.join(PATH,'label','label.txt'))

# K-Means para
NUM_CLUSTER = 5    
MAX_ITERS = 20

# Training para
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
TRAIN_STEPS = 500000
PIC_TYPE = 'jpg'             # the picture format of training images.
RESTORE_MODEL = False
MAX_NUM_GT = 10              # Suppose that each image contains up to 10 objects

# YOLOv2 para
CHANNEL = 3
INPUT_SIZE = 320
OBJECT_SCALE = 5
NOOBJECT_SCALE = 1
CLASS_SCALE = 1
COORD_SCALE =1
MAX_IOU = 0.6
CONFIDENCE_THRESHOLD = 0.1  # background when (confidence score * maximal prob )< 0.24.
NMS_THRESHOLD = 0.5
NUM_ANCHORS = NUM_CLUSTER    # YOLOv2 will predict NUM_CLUSTER bounding boxes in each grid cell.
DF = 32                      # downsampling factor
S = int(INPUT_SIZE/DF)       # divides the input image into an S x S grid.
NUM_CLASSESS = len(LABELS.Class_name)
OUYPUT_CHANNELS = NUM_ANCHORS*(4+1+NUM_CLASSESS)
assert INPUT_SIZE % DF==0

MODEL_NAME = 'model.ckpt'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(PATH,'model','checkpoint')
