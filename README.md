# Object Detection
This is the implementation of YOLOv2 for object detection in Tensorflow. It contains complete code for preprocessing, training and test. Besides, this repository is easy-to-use and can be developed on Linux and Windows.  

[YOLOv2 : Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.](https://arxiv.org/abs/1612.08242)

## Getting Started
### 1 Prerequisites  
* Python3.6  
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
`run python spilt.py`  
Then train and val images will be generated in  `../YOLOv2/data/annotation/train` and  `/YOLOv2/data/annotation/test` directories, respectively.  

### 4 Train model  
The model parameters, training parameters and eval parameters are all defined by `parameters.py`.  
`cd ../YOLOv2/Code`  
`run python train.py`  
The model will be saved in directory `../YOLOv2/model/checkpoint`, and some detection results are saved in `../YOLOv2/pic`. 
 
### 5 Visualize model using Tensorboard  
`cd ../YOLOv2`  
`run tensorboard --logdir=model/`   
Open the URL in browser to visualize model.  

## Examples  
Here are two successful detection examples in my dataset:   
