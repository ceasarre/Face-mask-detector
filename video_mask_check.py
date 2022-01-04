import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread
import matplotlib.pyplot as plt


MASK_CLASSIFIER_PATH = r'models/model_20220104-111420.h5'
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
mask_classiefier = load_model(MASK_CLASSIFIER_PATH)

# Constants
color_mask = (0, 255, 0)
color_no_mask = (0,0,255)

width_sec = 160
height_sec = 160
depth_sec = 3

# Text info
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1


class CheckMask:

    def __init__(self, frame = None) -> None:
        self.frame = frame
        self.face_classifier = faceCascade
        self.mask_classifier = mask_classiefier
        self.faces = 0
        self.mask = []
       
    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def show_frame(self):
        plt.imshow(self.frame)

    def detect_faces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_classifier.detectMultiScale(gray,
                                    scaleFactor=1.1,
                                    minNeighbors=8,
                                    minSize=(60, 60),
                                    flags=cv2.CASCADE_SCALE_IMAGE)


        # Debug info
        # print("Detected: {} faces".format(len(self.faces)))
        # print(self.faces)
    
    def check_if_mask(self):
        for (x,y,w,h) in self.faces:

            detected = np.zeros(shape=(w,h,3))
            detected = self.frame[y : y + h, x : x + w, :]
            # faces_detected.append(detected)
            # cv2.rectangle(image, (x, y), (x + w, y + h),(0,255,0), 2)

            # detect mask
            # resize image to the next classifier
            detected = cv2.resize(detected, (width_sec,height_sec))
            
            # add new dimension
            
            detected = detected[np.newaxis, ...]
            check_mask = np.argmax(mask_classiefier.predict(detected))
            
            if check_mask == 0:
                self.mask.append(True)
            else:
                self.mask.append(False)

    def analyze_frame(self):
        self.detect_faces()
        self.check_if_mask()