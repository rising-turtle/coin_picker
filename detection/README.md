# Coin detection using OpenCV with Jetson Nano and RealSense

## Getting Started

Python version 3.12.7

```sh
pip install opencv-python
# TODO: write requirements.txt
```

To take pictures with RealSense:

```sh
cd catkin_ws/src
roscore
python coin_picker/get_images/image_sub.py
roslaunch ros…/launch/rs_wrapper.launch
# change launch config file to include depth information
```


## Game Plan

Prerequisite: Learn how to optimize process for our Jetson Nano with [Jetson Inference](https://github.com/dusty-nv/jetson-inference)

1. Detect and localize the uppermost coins
    - OpenCV?
    - MobileNetV2-SSD: [Collect your own Detection Datasets](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md)
    - Manuever Mirobot
2. Segment the coins
    - rectangular: Use output from previous step as is
    - precise: OpenCV, or a segmentation model
3. Classify the coins
    - SVM
    - ResNet18: [Collect your own Classification Datasets](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md) (standard image size 224x224)

Compare output with every variation in the pipeline.


## Papers on Small Object Detection

- [ ] Unread
- [x] Read

- [x] SSD: Single Shot Multibox Detector ([arXiv](https://arxiv.org/pdf/1512.02325))
- [x] A Multi-Scale Traffic Object Detection Algorithm for Road Scenes Based on Improved YOLOv5 ([paper](https://doi.org/10.3390/electronics12040878))

- [ ] Slicing Aided Hyper Inference (SAHI) [GitHub](https://github.com/obss/sahi): A lightweight vision library for performing large scale object detection & instance segmentation
- [ ] [Small Object Detection using Context and Attention](https://arxiv.org/pdf/1912.06319) (paper)
- [ ] [The Power of Tiling for Small Object Detection](https://openaccess.thecvf.com/content_CVPRW_2019/papers/UAVision/Unel_The_Power_of_Tiling_for_Small_Object_Detection_CVPRW_2019_paper.pdf) (paper)

- [ ] Focal Loss for Dense Object Detection ([arXiv](https://arxiv.org/pdf/1708.02002))
- [ ] YOLO: You Only Look Once ([arXiv](https://arxiv.org/pdf/1506.02640))


## Papers on Circle Detection and Segmentation

- [x] `cv2.minEnclosingCircle`: [Smallest enclosing disks (balls and ellipsoids)](https://people.inf.ethz.ch/emo/PublFiles/SmallEnclDisk_LNCS555_91.pdf)
- [x] Cuevas, E., Wario, F., Osuna-Enciso, V., Zaldivar, D., Pérez-Cisneros, M. Fast algorithm for multiple-circle detection on images using learning automata, IET Image Processing 6 (8) , (2012), pp. 1124-1135 ([arXiv](https://arxiv.org/pdf/1405.5531))
