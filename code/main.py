from Detector import *

# IMAGE_PATH = "./data/images/test.jpg"
# image = cv2.imread(IMAGE_PATH)
# imageHeight, imageWidth = image.shape[:2]

VIDEO_PATH = 0
cap = cv2.VideoCapture(VIDEO_PATH)
(sucess, image) = cap.read()
videoWidth = image.shape[1]
videoHeight = image.shape[0]

ksize = 5
t1 = 50
t2 = 55

detector = Detector()

detector.detectVideo(VIDEO_PATH, ksize, t1, t2, videoHeight, videoWidth)

# lines = detector.preprocessing(image, ksize, t1, t2)
# image = detector.detectImage(lines, imageHeight, imageWidth)
# detector.saveImage(image, "result", True)
