import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
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
        self.stopped = False
       

    def show_frame(self) -> None:
        plt.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

    def detect_faces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_classifier.detectMultiScale(gray,
                                    scaleFactor=1.1,
                                    minNeighbors=8,
                                    minSize=(60, 60),
                                    flags=cv2.CASCADE_SCALE_IMAGE)



    
    def check_if_mask(self) -> None:
        for (x,y,w,h) in self.faces:

            detected = np.zeros(shape=(w,h,3))
            detected = self.frame[y : y + h, x : x + w, :]

            detected = cv2.resize(detected, (width_sec,height_sec))
            
            detected = detected[np.newaxis, ...]
            check_mask = np.argmax(mask_classiefier.predict(detected))
            
            if check_mask == 0:
                self.mask.append(True)
            else:
                self.mask.append(False)

    def add_mark(self) -> None:
        # Constants
        color_mask = (0, 255, 0)
        color_no_mask = (0,0,255)
        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2

        for i, (x,y,w,h) in enumerate(self.faces):
            
            if self.mask[i] is True:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color_mask, 2)
                cv2.putText(self.frame, 'Mask', (x - 10, y-20 ), font, 
                    fontScale, color_mask, thickness, cv2.LINE_AA)
            else:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color_no_mask, 2)
                cv2.putText(self.frame, 'NO MASK', (x - 10, y-20 ), font, 
                            fontScale, color_no_mask, thickness, cv2.LINE_AA)  


    def detect_mask(self):
        self.detect_faces()
        self.check_if_mask()
        self.add_mark()

        return self.frame
    

    # Use automaticly by Python garabage collector
    def __del__(self):
        pass
        # Debug LOG
        # print('Object destroyed')