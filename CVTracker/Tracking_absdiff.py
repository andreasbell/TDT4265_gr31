
# coding: utf-8

# In[1]:

import numpy as np
import csv
import cv2
import sys

def nothing(x):
    pass

def movementSearch(th, frame):
    objectDetected = False
    im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0: objectDetected = True
    
    if objectDetected:
        hull = cv2.convexHull(contours[-1])

        largest = contours[-1]
        #x,y,w,h = cv2.boundingRect(largest)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,255,0),2)
        #cv2.drawContours(frame,contours,-1,(0,255,0),2)
        
        
    else:
        print("no det")

        
cap = cv2.VideoCapture(0)

cv2.namedWindow("trackwin")
cv2.createTrackbar("track1","trackwin", 30, 100, nothing)

while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()
    if not ret:
       print('Cannot read webcam file')
       sys.exit()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY )
    
    ret, frame2 = cap.read()    
    if not ret:
        print('Cannot read webcam file')
        sys.exit()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY )
    
    sens = cv2.getTrackbarPos("track1","trackwin")
    
    #Get difference of the two consecutive frames and threshold
    frame_diff = cv2.absdiff(gray1,gray2)
    ret,th = cv2.threshold(frame_diff,sens,255,cv2.THRESH_BINARY)
    
    #Blur the thresholded image and re-threshold
    th = cv2.blur(th,(40,40))
    ret,th = cv2.threshold(th,sens,255,cv2.THRESH_BINARY)

    cv2.imshow('th', th)
    
    movementSearch(th,frame1)
    
    cv2.imshow("Box",frame1)

        
    # Exit if ESC pressed
    k = cv2.waitKey(17) & 0xff
    if k == 27 : break
    
cap.release()
cv2.destroyAllWindows()


# In[ ]:

import numpy as np
import csv
import cv2
import sys

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame1 = cap.read()
    if not ret:
       print('Cannot read webcam file')
       sys.exit()
    cv2.imshow("test",frame1)
    
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
        
cap.release()
cv2.destroyAllWindows()


# In[12]:

cap.release()
cv2.destroyAllWindows()


# In[ ]:



