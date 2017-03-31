import numpy as np
import csv
import cv2
import sys

class CVTracker():
    def __init__(self):
        self.prevFrame = None
        self.lower_range = (137,5,146)
        self.upper_range = (198,74,240)
        self.absdiff_thresh = 20

    def nothing(self):
        pass
    
    def movementSearch(self, thresh, frame):
        objectDetected = False
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0: objectDetected = True

        if objectDetected:
            hull = cv2.convexHull(contours[-1])

            largest = contours[-1]
            #x,y,w,h = cv2.boundingRect(largest)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            self.box = np.int0(box)
            cv2.drawContours(frame,[self.box],0,(0,255,0),2)
            #cv2.drawContours(frame,contours,-1,(0,255,0),2)

    def AD_tracker(self, frame):
        if type(self.prevFrame) == type(None):
            self.prevFrame = frame
            print("test")
      
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
        gray2 = cv2.cvtColor(self.prevFrame, cv2.COLOR_BGR2GRAY )

        #Get difference of the two consecutive frames and threshold
        frame_diff = cv2.absdiff(gray2,gray1)

        ret, thresh = cv2.threshold(frame_diff, self.absdiff_thresh, 255, cv2.THRESH_BINARY)

        #Blur the thresholded image and re-threshold
        thresh = cv2.blur(thresh,(40,40))
        ret, thresh = cv2.threshold(thresh, self.absdiff_thresh, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        self.thresh = thresh

        self.prevFrame = frame.copy()
        
        #Find bounding box
        self.movementSearch(thresh, frame)
        
        
    def HSV_tracker(self, frame):
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #Threshold for skin color
        thresh = cv2.inRange(HSV, self.lower_range, self.upper_range)
        
        #Erode and dilate the thresholded image
        #erode_size = 7
        #dilate_size = 8
        #erode_el = cv2.getStructuringElement(cv2.MORPH_RECT,(erode_size,erode_size))
        #dilate_el = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_size,dilate_size))

        #thresh = cv2.erode(thresh,erode_el)
        #thresh = cv2.erode(thresh,erode_el)
        #thresh = cv2.dilate(thresh,dilate_el)
        #thresh = cv2.dilate(thresh,dilate_el)
        
        #Or just blur it
        thresh = cv2.blur(thresh,(40,40))
        ret,thresh = cv2.threshold(thresh,30,255,cv2.THRESH_BINARY)    

        self.movementSearch(thresh,frame)
        
    def HSV_calibrate(self):
        pass
        
    def AD_calibrate(self):
        pass



