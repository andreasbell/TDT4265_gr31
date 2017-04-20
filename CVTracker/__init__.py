import numpy as np
import csv
import cv2
import sys

class CVTracker():
    def __init__(self):
        self.image = None
        self.lower_range = (137,5,146)
        self.upper_range = (198,74,240)
        self.absdiff_blur = 40     #Size of blurring kernel
        self.absdiff_thresh_1 = 30 #Threshold value before blurring
        self.absdiff_thresh_2 = 30 #Threshold value after blurring
        self.erode = 11
        self.calibrating = False
        self.y = 0

    def movementSearch(self, thresh, frame):
        objectDetected = False
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0: objectDetected = True

        if objectDetected:
            largest = max(contours, key = lambda x: cv2.contourArea(x))

            hull = cv2.convexHull(largest)
            #x,y,w,h = cv2.boundingRect(largest)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            rect = cv2.minAreaRect(largest)
            self.y = rect[0][1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,255,0),2)
            #cv2.drawContours(frame,contours,-1,(0,255,0),2)

    def AD_tracker(self, frame):
        if type(self.image) == type(None):
            self.image = frame
            print("test")
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
        gray2 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY )

        #Get difference of the two consecutive frames and threshold
        frame_diff = cv2.absdiff(gray2,gray1)
        
        cv2.imshow("fdiff",frame_diff)
        #self.frame_diff = frame_diff.copy()
        
        ret, thresh = cv2.threshold(frame_diff, self.absdiff_thresh_1, 255, cv2.THRESH_BINARY)
        erode_el = cv2.getStructuringElement(cv2.MORPH_RECT,(self.erode, self.erode))
        thresh = cv2.erode(thresh, erode_el)
        
        cv2.imshow("thresh", thresh)
        #self.thresh = thresh.copy()
        
        #Blur the thresholded image and re-threshold
        thresh = cv2.blur(thresh,(self.absdiff_blur, self.absdiff_blur))
        ret, thresh = cv2.threshold(thresh, self.absdiff_thresh_2, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        cv2.imshow("blur", thresh)
        
        #Keep the previous frame for next loop.
        self.image = frame.copy()
        #self.thresh = thresh.copy()
        
        #Find bounding box
        if not self.calibrating:
            self.movementSearch(thresh, frame)
        
        
    def HSV_tracker(self, frame):
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.image = frame.copy()
        thresh = 0
        
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
        self.thresh = thresh.copy()
        
        if not self.calibrating:
            self.movementSearch(thresh,frame)

        
    def HSV_calibrate(self):
        if not self.calibrating:
            cv2.namedWindow("HSV_calibrate")
            cv2.createTrackbar("Hmin","HSV_calibrate", self.lower_range[0], 180, self.HSV_slider_change)
            cv2.createTrackbar("Hmax","HSV_calibrate", self.upper_range[0], 180, self.HSV_slider_change)
            cv2.createTrackbar("Smin","HSV_calibrate", self.lower_range[1], 256, self.HSV_slider_change)
            cv2.createTrackbar("Smax","HSV_calibrate", self.upper_range[1], 256, self.HSV_slider_change)
            cv2.createTrackbar("Vmin","HSV_calibrate", self.lower_range[2], 256, self.HSV_slider_change)
            cv2.createTrackbar("Vmax","HSV_calibrate", self.upper_range[2], 256, self.HSV_slider_change)
            self.calibrating = True
        elif self.calibrating:
            cv2.destroyWindow("HSV_calibrate")
            self.calibrating = False
        
    def AD_calibrate(self):
        if not self.calibrating:
            cv2.namedWindow("AD_calibrate")
            cv2.createTrackbar("Thresh_1", "AD_calibrate", self.absdiff_thresh_1, 100, self.AD_slider_change)
            cv2.createTrackbar("Thresh_2", "AD_calibrate", self.absdiff_thresh_1, 100, self.AD_slider_change)
            cv2.createTrackbar("Blur",     "AD_calibrate", self.absdiff_blur,     100, self.AD_slider_change)
            cv2.createTrackbar("Erode",    "AD_calibrate", self.erode,            100, self.AD_slider_change)

            self.calibrating = True
        elif self.calibrating:
            cv2.destroyWindow("AD_calibrate")
            self.calibrating = False
            
            
    def HSV_slider_change(self, val):
        H_low  = cv2.getTrackbarPos("Hmin", "HSV_calibrate")
        H_high = cv2.getTrackbarPos("Hmax", "HSV_calibrate")
        S_low  = cv2.getTrackbarPos("Smin", "HSV_calibrate")
        S_high = cv2.getTrackbarPos("Smax", "HSV_calibrate")
        V_low  = cv2.getTrackbarPos("Vmin", "HSV_calibrate")
        V_high = cv2.getTrackbarPos("Vmax", "HSV_calibrate")
        self.lower_range = (H_low,S_low,V_low)
        self.upper_range = (H_high,S_high,V_high)

        
    def AD_slider_change(self, val):
        self.absdiff_thresh_1 = cv2.getTrackbarPos("Thresh_1", "AD_calibrate")
        self.absdiff_thresh_2 = cv2.getTrackbarPos("Thresh_2", "AD_calibrate")
        self.absdiff_blur     = cv2.getTrackbarPos("Blur",     "AD_calibrate") + 1
        self.erode            = cv2.getTrackbarPos("Erode",     "AD_calibrate") + 1

        



