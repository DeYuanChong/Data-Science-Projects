# Yolov3Tiny Object Detection using Tensorflow 2.0 with GPU support (Nvidia 940MX)

## Main Changes
1) Ran a Yolov3Tiny instead of Yolov3 due to memory constraints
2) Decreased confidence score threshold to 0.4 for yolov3tiny model, as the video is quite zoomed out and hard for the model to detect
3) Increased IOU threshold, as many players are overlapping and running past each other. (Want to keep the detections separate)


### Video example
![demo](https://github.com/DeYuanChong/Data-Science-Projects/blob/main/Yolov3%20Video%20Detection/detections/England%20v%20Italy.gif)

