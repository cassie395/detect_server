import numpy as np
import argparse
import time
import cv2
import os

confthres=0.5
nmsthres=0.4
yolo_path="./"

box_num = [0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0]
box_num2 = [0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0]

def get_labels(labels_path):
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def get_predection(image,net,LABELS,COLORS):

    box_num = [0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0]
    
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            #print(boxes)
            #print(classIDs[i])
            box_num[classIDs[i]] = box_num[classIDs[i]]+1
            box_num2[classIDs[i]] = box_num[classIDs[i]]
            #print(box_num[classIDs[i]])
            #print(text)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        for i in range(20):
            box_num2[i] = box_num[i]
        
        print(box_num)
    return image
    

def runModel(image):
    
    labelsPath="model_data/med.names"
    cfgpath="model_data/yolo-obj-med.cfg"
    wpath="model_data/yolo-obj-med.weights"
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)
    res=get_predection(image,nets,Lables,Colors)
    
    return box_num2