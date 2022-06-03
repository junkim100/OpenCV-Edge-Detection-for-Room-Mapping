import cv2
import matplotlib.pyplot as plt
import time

class Detector:
    def __init__(self):
        pass

    def detect(self, img, t1, t2):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        canny = cv2.Canny(blur, threshold1=t1, threshold2=t2)
        # plt.imshow(canny)
        return canny

    def detectVideo(self, videoPath, t1, t2):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Error opening file...")
            return
        
        (sucess, image) = cap.read()

        startTime = 0

        while sucess:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            edge = self.detect(image, t1, t2)

            cv2.putText(edge, "FPS: " + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Result", edge)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            (sucess, image) = cap.read()
        
        cv2.destroyAllWindows()