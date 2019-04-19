from ctypes import *
from sort import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


def cvDrawBoxes(detections, img):
    ide = 0
    for detection in detections:
        xmin, ymin, xmax, ymax = detection[0],detection[1],detection[2],detection[3]
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img, str(ide) +
                    detection[4][0].decode() +
                    " [" + str(round(detection[4][1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        ide += 1
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("samples/001.wmv")
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "outputs/001.avi", cv2.VideoWriter_fourcc(*"MJPG"), 24.0,
        (640, 480))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    # Video default size (640x480)
    darknet_image = darknet.make_image(640,480,3)

    #Inicializar o Sort
    mot_tracker = Sort()
    while True:
        
        prev_time = time.time()
        ret, frame_read = cap.read()
        if frame_read is None:
            break
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (640,480),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        print(detections)
        #Sort Track update
        track_bbs_ids = mot_tracker.update(detections)

        image = cvDrawBoxes(track_bbs_ids, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
