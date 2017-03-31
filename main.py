from DLClassifier import DLClassifier
from CVClassifier import CVClassifier
import numpy as np
import cv2

#det = DLClassifier("weights/detector.ckpt")
det = CVClassifier()

np.set_printoptions(precision=1, suppress=True)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #Detect objects
    det.detect_from_cvmat(frame)
    #res = det.show_results(frame,yolo.result)
    #print(det.result)
    print(str(det.result) + " "*20, sep=' ', end='\r', flush=True)

    # Display the resulting frame
    #cv2.imshow('frame: ',frame)
    cv2.imshow('frame: ',det.image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

