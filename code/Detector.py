from hmac import new
from re import I
from tkinter.tix import DirList
from turtle import shape
import cv2
from cv2 import HoughLines
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from itertools import combinations

class Detector:
    def __init__(self):
        self.intersection = np.empty(shape=(2))

    def preprocessing(self, image, ksize, t1, t2):
        # Preprocessing and Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        canny = cv2.Canny(blur, threshold1=t1, threshold2=t2)

        # Process edge image
        kernel = np.ones((15,15),np.uint8)
        edge = cv2.dilate(canny, kernel, iterations=1)

        # edge = self.continuousPixel(self.pixelValue(edge),20)
        # edge = self.continuousPixel(self.pixelValue(edge),20)
        # edge = self.continuousPixel(self.pixelValue(edge),20)
        # edge = self.continuousPixel(self.pixelValue(edge),10)
        # edge = self.continuousPixel(self.pixelValue(edge),5)

        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)
        
        lines = cv2.HoughLinesP(
            edge, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=500, # Min allowed length of line
            maxLineGap=100 # Max allowed gap between line for joining them
            )

        return lines

    def findIntersection(self, slope, yIntercept, width, height, showPlot):
        x = np.arange(width)
        Y = np.empty(shape=(len(slope), width))
        f = np.empty(width)

        # Find the equation f for each line and save it to Y
        for i in range(len(slope)):
            for j in range(width):
                f[j] = slope[i]*j + yIntercept[i]
            Y[i] = f
            plt.plot(x, Y[i])
            f = np.empty(width)

        # Find intersections
        comb = combinations(range(len(Y)), 2)
        interpoint = []
        xInter = 0
        yInter = 0
        for i, j in comb:
            idx = np.argwhere(np.diff(np.sign(Y[i] - Y[j]))).flatten()
            if len(idx) == 0:
                continue
            for k in idx:
                xInter += k
                yInter += Y[i][k]
        xInter = xInter/(len(Y))
        yInter = yInter/(len(Y))

        # Save intersection to private variable intersection
        self.intersection[0] = xInter
        self.intersection[1] = yInter

        # Plot intersections
        plt.plot(self.intersection[0], self.intersection[1], 'ro', markersize=3)

        # Change axis to represent the image
        plt.ylim(0, height)
        plt.gca().invert_yaxis()

        if showPlot:
            plt.show()

    def filterLines(self, linesList, thresholdOverlap, thresholdSlope, thresholdYIntercept, width, height, showPlot):
        # Remove overlapping lines
        for line in linesList:
            for line2 in linesList:
                if line == line2:
                    continue
                elif abs(line[0] - line2[0]) < thresholdOverlap and abs(line[1] - line2[1]) < thresholdOverlap and abs(line[2] - line2[2]) < thresholdOverlap and abs(line[3] - line2[3]) < thresholdOverlap:
                    linesList.remove(line2)

        # Get slope and y-intercept for each line
        slope = []
        yIntercept = []
        for line in linesList:
            if line[0] == line[2]:
                slope.append(0)
                yIntercept.append(line[1])
            else:
                s = (line[1]-line[3])/(line[0]-line[2])
                slope.append(s)
                yIntercept.append(s*(-1*line[0]) + line[1])

        # Remove lines with similar slope and y-intercept
        index = []
        for i in range(len(slope)):
            for j in range(len(slope)):
                if i == j:
                    continue
                elif abs(slope[i] - slope[j]) < thresholdSlope and abs(yIntercept[i] - yIntercept[j]) < thresholdYIntercept:
                    index.append(j)
        
        # Remove duplicates
        temp = []
        for i in index:
            if i not in temp:
                temp.append(i)
        index = temp
        index.sort()

        newIndex = []
        for i in range(len(linesList)):
            newIndex.append(i)
        for i in index:
            newIndex.remove(i)

        # Final filter lists
        newLinesList = []
        newSlope = []
        newYIntercept = []
        for i in newIndex:
            newLinesList.append(linesList[i])
            newSlope.append(slope[i])
            newYIntercept.append(yIntercept[i])

        self.findIntersection(newSlope, newYIntercept, width, height, showPlot)

        return newLinesList

    def isCenter(self, height, width, threshold):
        if len(self.intersection) == 0:
            return

        # Intersection at center
        if abs(self.intersection[0] - width/2) < threshold and abs(self.intersection[1] - height/2) < threshold:
            return 0
        # Intersection at left top corner
        elif self.intersection[0] < width/2 and self.intersection[1] < height/2:
            return 1
        # Intersection at left bottom corner
        elif self.intersection[0] < width/2:
            return 2
        # Intersection at right top corner
        elif self.intersection[0] > width/2 and self.intersection[1] < height/2:
            return 3
        # Intersection at right bottom corner
        elif self.intersection[0] > width/2:
            return 4
        # Error
        else:
            return -1

    def showLines(self, lines, height, width, image):
        # Create linesList
        linesList = []
        for points in lines:
            x1,y1,x2,y2=points[0]
            linesList.append([x1,y1,x2,y2])

        # Filter lines
        linesList = self.filterLines(linesList, 600, 0.1, 0, width, height, False)
        linesList = self.filterLines(linesList, 600, 0.1, 0, width, height, False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = "Center found"
        text2 = "Please move the corner to the center of the image"
        text1size = cv2.getTextSize(text1, font, 1, 2)[0] 
        text2size = cv2.getTextSize(text2, font, 1, 2)[0]
        text1X = int((width - text1size[0]) / 2)
        text2X = int((width - text2size[0]) / 2)

        move = self.isCenter(height, width, 100)
        if move == 0:
            cv2.putText(image, text1, (text1X, 40), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(image, text2, (text2X, 40), font, 1, (255, 255, 255), 2)

        # Plot lines and intersection
        for line in linesList:
            # print(line)
            cv2.line(image,(line[0],line[1]),(line[2],line[3]),(255,255,255),2)
        
        if len(self.intersection) != 0:
            cv2.circle(image, (int(self.intersection[0]), int(self.intersection[1])), 10, (0, 0, 255), -1)

        return image

    def detectVideo(self, videoPath, ksize, t1, t2, height, width):

        # Used for FPS
        startTime = 0

        # Open video file
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened() == False):
            print("Error opening file...")
            return

        # Show Video
        while cap.isOpened():
            (sucess, image) = cap.read()
            if sucess:
                empty = np.zeros(shape=(height, width, 3))

                currentTime = time.time()
                fps = 1 / (currentTime - startTime)
                startTime = currentTime

                cv2.putText(image, "FPS: " + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                lines = self.preprocessing(image, ksize, t1, t2)
                if lines is None:
                    print("{}\t\tNo lines detected".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(currentTime))))
                    cv2.imshow("result", image)
                else:
                    print("{}\t\tLines detected".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(currentTime))))
                    image = self.showLines(lines, height, width, image)
                    cv2.imshow("result", image)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
            else:
                break

        cv2.destroyAllWindows()

    def detectImage(self, lines, height, width):
        empty = np.zeros(shape=(height, width, 3))

        if lines is None:
            print("No lines detected")
            return
        return self.showLines(lines, height, width, empty)

    def saveImage(self, image, name, save):
        cv2.imshow(name, image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        if save:
            cv2.imwrite("./data/images/{}.jpg".format(name), image)
