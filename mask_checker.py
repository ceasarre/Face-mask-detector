import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread
from check_mask import CheckMask

class Analyzer:

    def __init__(self, frame) -> None:

        self.mask_checker = CheckMask(frame)
        self.frame = self.mask_checker.frame
        self.stopped = False

    def start(self):
        Thread(target=self.analyze, args=()).start()
        return self
    
    def stop(self):
        self.stopped = True

    def analyze(self):
        while not self.stopped:
            self.mask_checker.detect_faces()
            self.frame = self.mask_checker.frame

            # Debug Log
            print('ANALYZE...')