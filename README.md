# Object Detection
This is the implementation of YOLOv2 for object detection in Tensorflow. It contains complete code for preprocessing, training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows.  

[YOLOv2 : Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1612.08242)

## Getting Started
### 1 Prerequisites  
* Python3  
* Tensorflow  
* Opencv-python  
* Pandas  

### 2 Define your class names  
Download  and unzip this repository.  
`cd ../YOLOv2/label`  
Open the `label.txt` and revise its class names as yours.  

### 3 Prepare images  
Copy your images and annotation files to directories `../YOLOv2/data/annotation/images` and `../YOLOv2/data/annotation/images/xml` respectively, where the annotations should be obtained by [a graphical image annotation tool](https://github.com/tzutalin/labelImg) and  saved as XML files in PASCAL VOC format.  
`cd ../YOLOv2/Code`  
run  
`python spilt.py`  
Then train and val images will be generated in  `../YOLOv2/data/annotation/train` and  `/YOLOv2/data/annotation/test` directories, respectively.  

### 4 Anchor clusters using K-means  
Run K-means clustering on the training set bounding boxes to automatically find good anchors.  
`cd ../YOLOv2/Code`  
run  
`python anchor_cluster.py`  
Anchors generated by K-means are saved the directory `../YOLOv2/anchor/anchor.txt`. Belows are same outputs when running K-means:

Iter = 1/20, Average IoU = 0.72583, is current optimal anchors.  
Iter = 2/20, Average IoU = 0.732365, is current optimal anchors.  
Iter = 3/20, Average IoU = 0.734656, is current optimal anchors.  
Iter = 4/20, Average IoU = 0.735472, is current optimal anchors.  
Iter = 5/20, Average IoU = 0.735702, is current optimal anchors.  
Iter = 6/20, Average IoU = 0.735694  
Iter = 7/20, Average IoU = 0.735552  
Iter = 8/20, Average IoU = 0.735343  
Iter = 9/20, Average IoU = 0.735099  
Iter = 10/20, Average IoU = 0.734816  
Iter = 11/20, Average IoU = 0.734603  
Iter = 12/20, Average IoU = 0.734358  
Iter = 13/20, Average IoU = 0.734128  
Iter = 14/20, Average IoU = 0.733794  
Iter = 15/20, Average IoU = 0.73353  
Iter = 16/20, Average IoU = 0.733289  
Iter = 17/20, Average IoU = 0.7331  
Iter = 18/20, Average IoU = 0.732887  
Iter = 19/20, Average IoU = 0.732687  
Iter = 20/20, Average IoU = 0.732539  

### 5 Train model using Tensorflow  
The model parameters, training parameters and eval parameters are all defined by `parameters.py`.  
`cd ../YOLOv2/Code`  
run  
`python train.py`  
The model will be saved in directory `../YOLOv2/model/checkpoint`, and some detection results are saved in `../YOLOv2/pic`. 
 
### 6 Visualize model using Tensorboard  
`cd ../YOLOv2`  
run  
`tensorboard --logdir=model/`   
Open the URL in browser to visualize model.  

## Examples  
Here are two successful detection examples in my dataset:   
