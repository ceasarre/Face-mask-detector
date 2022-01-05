import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread
from check_mask import CheckMask

class MaskAnalyzer:

    def __init__(self, frame = None) -> None:

        self.frame = frame
        self.stopped = False
        self.isVideoAvaiable = False

    def start(self):
        Thread(target=self.analyze, args=()).start()
        
        # Debug info:
        print("ANALYZE THREAD STARTED")
        self.isVideoAvaiable = True
        
        return self
    
    def stop(self):
        self.stopped = True

    def analyze(self):
        while not self.stopped:
            
            if self.isVideoAvaiable:
                checker = CheckMask(self.frame)
                self.frame = checker.detect_mask()

                # Debug Log
                # print('ANALYZED...')