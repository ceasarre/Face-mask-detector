import os
import cv2
from video_get import VideoGet
from video_show import VideoShow
import argparse

def thread_video_get(source = 0):
    # Thread for grabbing video frames with VideoGet object

    video_getter = VideoGet(source).start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        cv2.imshow('Video', frame)

def mulithread_processing(source = 0):

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    
    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break
        
        frame = video_getter.frame
        video_shower.frame = frame

def main():
    mulithread_processing(source=0)

if __name__ == "__main__":
    main()