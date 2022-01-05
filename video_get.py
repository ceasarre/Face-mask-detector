from threading import Thread
import cv2

class VideoGet:


    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        # Debug info:   
        print("GET THREAD STARTED")
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                
            else:
                (self.grabbed, self.frame) = self.stream.read()
                # print('get')
                

    def stop(self):
        self.stopped = True