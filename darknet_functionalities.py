import cv2
import numpy as np


def buildObjectDetected(class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    label = str(classes[class_id])
    obj={    }
    obj["class"]=label
    obj["confidence"]=confidence
    obj["x"]=x
    obj["y"]=y
    obj["height"]=y_plus_h
    obj["width"]=x_plus_w


    return obj


# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def do_image_processing(net,classes, imagePassed):
    """
    docstring
    """
    # read input image
    image = cv2.imread(imagePassed)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392



    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)



    # run inference through the network
    # and gather predictions from output layers
    output_layers=get_output_layers(net)
    
    outs = net.forward(output_layers)
   
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    #List to save and return the response
    response=[]

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        #draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes, COLORS)
        newObj=buildObjectDetected(class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes)
        response.append(newObj)

    # display output image    
    #cv2.imshow("object detection", image)

    cv2.destroyAllWindows()
    return response

def loadClasses(classesPath):
    with open(classesPath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        return classes

def loadNet(weightsPath, configPath):
    # reading weights and its corresponding config file
    net = cv2.dnn.readNet(weightsPath, configPath)

    return net