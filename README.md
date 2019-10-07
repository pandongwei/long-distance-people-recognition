# long-distance-people-recognition

## Motivation

Robust people recognition with the combination of face recognition(MTCNN + Arcface), people detection(Yolov3) and kalman filter.

Since traditional face recognition have many questions,  i find it would be better, if i cound combine people detection and face recognition. Combining Kalman Filter can make the result much more stable.


## Quick Start

1.Download YOLOv3 or tiny_yolov3 weights from YOLO website. 

2.Install all requirements using command **sudo pip3 install -r requirements.txt**

3.Run command **python3 get_save_features.py --name ‘your name’** to record face in advance.

4.run the code using **python3 long_distance_people_recognition_with_kalman_filter.py**. (This code only use the webcam)
There has other version of kalman filter named deep sort, and it performs a little bit better but slower. I also realized this kind of version in ** long_distance_people_recognition_with_deep_sort.py**. To run it, you need to install tensorflow-gpu cause it needs tensorflow.

5.If you want to use realsense camera, run command ** python3 realsense_long_distance_people_recognition_with_kalman_filter.py**, you can see both RGB and depth image	

6.Then you can see the result.

## Output

### People recognition with confirmation:

![image](https://github.com/pandongwei/long-distance-people-recognition/blob/master/output/output1.gif)

Transfer the recognition result after the confirmation part and cancel the face recognition.


### Combination with kalman filter:

![image](https://github.com/pandongwei/long-distance-people-recognition/blob/master/output/output2.gif) 

After combining kalman filter people tracking part is robuster than before.

![image](https://github.com/pandongwei/long-distance-people-recognition/blob/master/output/output3.gif) 

Use Realsense D435 camera as input, so that distance from people to camera can be messured.
Output from  "realsense_long_distance_people_recognition_with_kalman_filter.py"

### TODO: sepcialised people following of the robot

## Dependencies

## Note
   This repository is based on some reference projects:
   
 https://github.com/shoutOutYangJie/Face_Recognition_using_pytorch
 
 https://github.com/ayooshkathuria/pytorch-yolo-v3
 
 https://github.com/nwojke/deep_sort
 
 https://github.com/srianant/kalman_filter_multi_object_tracking
