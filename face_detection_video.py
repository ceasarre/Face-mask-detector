from mtcnn import MTCNN
import cv2
import time



if __name__ == '__main__':

    detector = MTCNN()
    FRAME_RATE = 10
    PREV = 0

    # capture video
    video_cap = cv2.VideoCapture(0)

    # analyze video:
    while True:

        time_elapsed = time.time() - PREV
        ret, frame = video_cap.read()

        if time_elapsed > 1./FRAME_RATE:
            prev = time.time()
            frame = cv2.resize(frame, (600,400))
        
        # analyze if there is face in the frame
            boxes = detector.detect_faces(frame)
            if boxes:

        #     # save box
                box = boxes[0]['box']
                conf = boxes[0]['confidence']

        #     # cordinates, width and height
                x, y, width, height = box[0], box[1], box[2], box[3]

        #     # draw rectangle if face detected
                if conf > 0.5:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0,255,0), 1)
            
            cv2.imshow("Face detection", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
    video_cap.release()
    cv2.destroyAllWindows()
        