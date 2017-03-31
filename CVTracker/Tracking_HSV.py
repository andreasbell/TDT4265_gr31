
# coding: utf-8

# In[75]:

import numpy as np
import csv
import cv2
import sys


def slider_change():
    pass

def slider_update():
    H_low  = cv2.getTrackbarPos("Hmin","trackwin")
    H_high = cv2.getTrackbarPos("Hmax","trackwin")
    S_low  = cv2.getTrackbarPos("Smin","trackwin")
    S_high = cv2.getTrackbarPos("Smax","trackwin")
    V_low  = cv2.getTrackbarPos("Vmin","trackwin")
    V_high = cv2.getTrackbarPos("Vmax","trackwin")
    
    lower_range = (H_low,S_low,V_low)
    upper_range = (H_high,S_high,V_high)
    return lower_range, upper_range

def HSV_rectangle(crop):
    H_low  = int(crop[:,:,0].min())
    H_high = int(crop[:,:,0].max())
    S_low  = int(crop[:,:,1].min())
    S_high = int(crop[:,:,1].max())
    V_low  = int(crop[:,:,2].min())
    V_high = int(crop[:,:,2].max())
    lower_range = (H_low,S_low,V_low)
    upper_range = (H_high,S_high,V_high)
    return lower_range, upper_range


drawing = False # true if mouse is pressed
ix,iy = -1,-1
ex,ey = -1,-1
rect_selected = False

def draw_rectangle(event,x,y,flags,param):
    global ex,ey,ix,iy,drawing,rect_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_selected = False
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),-1)
            

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),-1)
        ex,ey = x,y
        rect_selected = True



def movementSearch(th, frame):
    objectDetected = False
    im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0: objectDetected = True
    
    if objectDetected:
        hull = cv2.convexHull(contours[-1])

        largest = contours[-1]
        #x,y,w,h = cv2.boundingRect(largest)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,255,0),2)
        #cv2.drawContours(frame,contours,0,(0,255,0),2)
        
        
cap = cv2.VideoCapture(0)

cv2.namedWindow("trackwin")
cv2.createTrackbar("Hmin","trackwin", 137, 256, slider_change)
cv2.createTrackbar("Hmax","trackwin", 198, 256, slider_change)
cv2.createTrackbar("Smin","trackwin", 5, 256, slider_change)
cv2.createTrackbar("Smax","trackwin", 74, 256, slider_change)
cv2.createTrackbar("Vmin","trackwin", 146, 256, slider_change)
cv2.createTrackbar("Vmax","trackwin", 240, 256, slider_change)
k = ord('s')
mode = "slider"
cv2.namedWindow("frame")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.equalizeHist(frame)
    if not ret:
       print('Cannot read webcam file')
       sys.exit()
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb )
    if k == ord('s'):
        mode = "slider"
    if mode == "slider":
        lower_range, upper_range = slider_update()        
    if k == ord('p'):
        crop = 0
        R = cv2.selectROI("frame", frame, True, False)
        mode = "ROI"
        x1 = int(R[0])
        x2 = int(R[0] + R[2])
        y1 = int(R[1])
        y2 = int(R[1] + R[3])
        crop = HSV[y1:y2,x1:x2]
        print(x1,x2,y1,y2)
        lower_range, upper_range = HSV_rectangle(crop)
        print(lower_range)
        print(upper_range)
        cv2.imshow("crop",crop)
        
    thresh = cv2.inRange(HSV, lower_range, upper_range)
    
    erode_size = 7
    dilate_size = 8
    erode_el = cv2.getStructuringElement(cv2.MORPH_RECT,(erode_size,erode_size))
    dilate_el = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_size,dilate_size))
    

    #thresh = cv2.erode(thresh,erode_el)
    #thresh = cv2.erode(thresh,erode_el)
    #thresh = cv2.dilate(thresh,dilate_el)
    #thresh = cv2.dilate(thresh,dilate_el)
    #thresh = cv2.blur(thresh,(40,40))
    #ret,thresh = cv2.threshold(thresh,30,255,cv2.THRESH_BINARY)    
    
    
    cv2.imshow('morphed', thresh)
    movementSearch(thresh,frame)
    
    cv2.imshow("frame",frame)
        
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
cap.release()
cv2.destroyAllWindows()


# In[74]:

cap.release()
cv2.destroyAllWindows()


# In[70]:

import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)


# In[ ]:



