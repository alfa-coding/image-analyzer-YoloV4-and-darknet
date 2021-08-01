import cv2
from darknet_functionalities import loadClasses, loadNet,do_image_processing

def test_CNNetworkload():
    newnet= loadNet("assets/yolov4-tiny.weights","assets/yolov4-tiny.cfg")
    assert newnet !=None

def test_loadClasses():
    assert len(loadClasses("assets/coco.names"))==80

def test_objectDetectionOverimage():
    newnet= loadNet("assets/yolov4-tiny.weights","assets/yolov4-tiny.cfg")
    classes= loadClasses("assets/coco.names")
    img= cv2.imread("assets/catANDdog.jpg")
    dataCollected = do_image_processing(newnet,classes,img)
    expected=[{'class': 'cat', 'confidence': 0.8230796456336975, 'x': 67, 'y': 66, 'height': 912, 'width': 731}, {'class': 'dog', 'confidence': 0.7829499244689941, 'x': 555, 'y': 8, 'height': 860, 'width': 1407}]
    assert expected ==dataCollected
