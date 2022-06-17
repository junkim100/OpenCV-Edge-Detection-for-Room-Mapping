from Decector import *

IMAGE_PATH = "./data/images/test.jpg"
image = cv2.imread(IMAGE_PATH)
imageHeight, imageWidth = image.shape[:2]

ksize = 5
t1 = 50
t2 = 55

detector = Detector()

lines = detector.preprocessing(image, ksize, t1, t2)
image = detector.detectImage(lines, imageHeight, imageWidth)
detector.saveImage(image, "result", True)
