
# coding: utf-8

# In[1]:

import cv2
import sys
 
if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    cap = cv2.VideoCapture(0)
 
    tracker = cv2.Tracker_create("MIL")
 
    # Read video
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Cannot read webcam file')
            sys.exit()
        cv2.imshow("Choose ROI press o", frame)
        
        k = cv2.waitKey(1) & 0xff
        if k == ord('o'):
            bbox = cv2.selectROI(frame, False)
            break
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
 
    while True:
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            break
         
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




# In[ ]:




# In[ ]:



