from video_mask_check import CheckMask
import cv2

IMAGE_PATH = r'res/couple.jpg'

image = cv2.imread(IMAGE_PATH)
image = cv2.resize(image, (1200,800))


mask = CheckMask(image)
mask.analyze_frame()
print(mask.mask)
print(mask.faces)

