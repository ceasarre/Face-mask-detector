import os
import cv2
from video_get import VideoGet
from video_show import VideoShow
from check_mask import CheckMask
import argparse

def mulithread_processing(source = 0):

    video_getter = VideoGet(source).start()
    # video_analyzer = CheckMask(video_getter.frame).start()
    video_shower = VideoShow(video_getter.frame).start()
    
    # checker = CheckMask()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            # video_analyzer.stop()
            video_getter.stop()
            break
        
        frame = video_getter.frame
        # checker.frame = frame
        # checker.detect_mask()
        checker = CheckMask(frame)
        video_shower.frame = checker.detect_mask()
        # del checker

def main():
    mulithread_processing(source=0)

if __name__ == "__main__":
    main()