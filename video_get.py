from threading import Thread
import cv2

class VideoGet:

    isVideoavaiable = False

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                VideoGet.isVideoavaiable = False
            else:
                (self.grabbed, self.frame) = self.stream.read()
                # print('get')
                VideoGet.isVideoavaiable = True

    def stop(self):
        self.stopped = True