import numpy as np
import cv2
import math

class CVClassifier:
    
    def __init__(self, invert = False):
        self.invert = invert
        pass
    
    def extract_hand(self, img):
        self.image = img.copy()
        
        # Convert to grayscale 
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Blurr image
        self.blurred = cv2.GaussianBlur(self.grey, (35, 35), 0)
        
        # Threshold using otsus method
        if self.invert == True:
            _, self.thresh = cv2.threshold(self.blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        else:
            _, self.thresh = cv2.threshold(self.blurred, 127, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Find contours, ad extract longest contour
        image_contours, contours, hierarchy = cv2.findContours(self.thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
    
        # Find Bounding box
        self.x, self.y ,self.w ,self.h = cv2.boundingRect(cnt)
        cv2.rectangle(self.image,(self.x, self.y),(self.x + self.w, self.y + self.h),(0,0,255),0)
    
        # Find convex hull 
        hull = cv2.convexHull(cnt)
        
        # Draw hull and contours
        self.drawing = np.zeros(self.image.shape,np.uint8)
        cv2.drawContours(self.drawing,[cnt],0,(0,255,0),0)
        cv2.drawContours(self.drawing,[hull],0,(0,0,255),0)
        
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        
        # Extract fingers
        self.count_defects = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                self.count_defects += 1
                cv2.circle(self.image,far,5,[0,0,255],-1)
            #dist = cv2.pointPolygonTest(cnt,far,True)
            #cv2.line(crop_img,start,end,[0,255,0],2)
            #cv2.circle(crop_img,far,5,[0,0,255],-1)
        return self.count_defects
    
    def detect_from_cvmat(self, img):
        self.extract_hand(img)
        if self.count_defects < 2:
            self.result = 0
        elif self.count_defects < 4:
            self.result = 1
        else:
            self.result = 2
        return self.result

if __name__ == "__main__":
    det = CVClassifier()
    cap = cv2.VideoCapture(0)
    np.set_printoptions(precision=1, suppress=True)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        det.detect_from_cvmat(frame)
        print(str(det.result) + " "*2, sep=' ', end='\r', flush=True)
    
        # Display the resulting frame
        cv2.imshow('frame: ', det.image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()