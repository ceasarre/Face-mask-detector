import os
import cv2
from video_get import VideoGet
from video_show import VideoShow
from check_mask import CheckMask
from mask_analyzer import MaskAnalyzer
import argparse

def mulithread_processing(source = 0):

    video_getter = VideoGet(source).start()
    # mask_analyzer = MaskAnalyzer(video_getter.frame).start()
    video_shower = VideoShow(video_getter.frame).start()
    
    # checker = CheckMask()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            # mask_analyzer.stop()
            video_getter.stop()

            break
        
        # Without multithread mask calculation
        frame = video_getter.frame
        checker = CheckMask(frame)
        video_shower.frame = checker.detect_mask()
        
        # Multithread processing
        # frame = video_getter.frame
        # # mask_analyzer.frame = frame
        # # video_shower.frame = mask_analyzer.frame
        # video_shower.frame = frame

def main():
    mulithread_processing(source=0)

if __name__ == "__main__":
    main()