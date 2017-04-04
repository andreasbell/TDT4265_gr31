import tkinter as tk
from PIL import Image
from PIL import ImageTk
from Game import peakdetect
import numpy as np
import time
import random
import cv2

from DLClassifier import DLClassifier
from CVClassifier import CVClassifier

class RPS_Game():
    def __init__(self):
        # GUI variabels
        self.opts = ["Tracking","Classic"]
        self.username = "Ketil"
        self.initMainMenu()      
        # GAME Countdown variabels
        
        self.difficultNames = ["Torstein", "SlickT", "Slick-T","Slick T", "Skorstein", "Bart", "Chilly cheese"]
        self.easyNames = ["Frank","Vipul"]
        
        self.wins = 0;
        self.losses = 0;
        self.draws = 0;
        
    def btnPress(self, press):
        print(press)
        if press == "Motion":
            self.opts[0] = "Motion" 
        elif press == "HSV":
            self.opts[0] = "HSV" 
        elif press == "Tracking":
            self.opts[0] = "Tracking" 
        elif press == "Neural":
            self.opts[1] = "Neural"
        elif press == "Classic":
            self.opts[1] = "Classic"
        elif press == "Play":
            self.username = self.entUsername.get()
            if self.username == "":
                print("Enter username!")
            elif self.opts[0] == "" or self.opts[1] == "":
                print("Select methods")
            else:
                print("LETS ROCK!")
                self.Play()
   
    def initMainMenu(self):
        # Initialize window and header image
        self.window = tk.Tk()
        self.window.title("Rock Paper Scissors")
        #window.geometry("300x300")
        #window.wm_iconbitmap("data_simple")
        RPS = cv2.imread("rps.png")
        RPS = cv2.cvtColor(RPS,cv2.COLOR_BGR2RGB)
        RPS = Image.fromarray(RPS)
        RPS = ImageTk.PhotoImage(RPS)
        headerImg = tk.Label( image = RPS)
        headerImg.image = RPS
        headerImg.pack()

        # Creating labels and entries
        lblTracking  = tk.Label(self.window, text = "Tracking method")
        lblDetection = tk.Label(self.window, text = "Detection method")
        lblUsername  = tk.Label(self.window, text = "Enter username: ")
        self.entUsername  = tk.Entry(self.window)
        self.entUsername.insert(0, "Vipul")

        # Create buttons
        btn_motion   = tk.Button(text = "Motion",   command = lambda press="Motion": self.btnPress(press))
        btn_HSV      = tk.Button(text = "HSV",      command = lambda press="HSV": self.btnPress(press))
        btn_tracking = tk.Button(text = "Tracking", command = lambda press="Tracking": self.btnPress(press))
        btn_neural   = tk.Button(text = "Neural",   command = lambda press="Neural": self.btnPress(press))
        btn_classic  = tk.Button(text = "Classic",  command = lambda press="Classic": self.btnPress(press))
        btn_play     = tk.Button(text = "Play",     command = lambda press="Play": self.btnPress(press))

        # Place text and buttons
        lblUsername.pack()
        self.entUsername.pack()
        lblTracking.pack(side=tk.LEFT)
        lblDetection.pack(side=tk.RIGHT)
        btn_motion.pack(side=tk.LEFT)
        btn_HSV.pack(side=tk.LEFT)
        btn_tracking.pack(side=tk.LEFT)
        btn_neural.pack(side=tk.RIGHT)
        btn_classic.pack(side=tk.RIGHT)
        btn_play.pack()
        
    def selectHand(self, hand, difficulty = 1):
        if difficulty == 0:
            #return loosing hand
            return (hand + 2)%3
        elif difficulty == 1:
            #return random hand
            return random.randint(0, 2)
        elif difficulty == 2:
            #return wining hand
            return (hand + 1)%2
    
    def countDown(self, hand_position = None, start = False):
        if start == True: # Start the count down
            self.start_time = int(round(time.time() * 1000)) #Time in ms
            self.hand_time = int(round(time.time() * 1000))
            self.hand_pos = []
        
        # Use timer if no hand position is given
        if hand_position == None:
            return (int(round(time.time() * 1000)) - self.start_time)//1000 #return seconds since start
        else:
            if int(round(time.time() * 1000)) - self.start_time > 3000:
                self.start_time = int(round(time.time() * 1000)) #Time in ms
                self.hand_time = int(round(time.time() * 1000))
                self.hand_pos = []
                
            current_time = int(round(time.time() * 1000))
            if current_time - self.hand_time > 100:
                self.hand_pos.append(hand_position)
                self.hand_time = current_time
            maxtab, mintab = peakdetect.peakdet(self.hand_pos,100)
            if(mintab > 0):
                self.start_time = int(round(time.time() * 1000)) #Time in ms
                self.hand_time = int(round(time.time() * 1000))
            return len(mintab)
            
            
        
    def run(self):
        self.window.mainloop()
        
    def display(self, frame, countdown, hand = None, result = "Draw"):
        if(hand is not None):
            #display hand
            img = cv2.imread('rps.png')
            size = np.shape(img)[1]
            img1 = cv2.resize(img[:, hand*size//3:(hand+1)*size//3],(448, 448))
            cv2.putText(img1,result, (3,3), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        else:
            #display countdown
            countdown_images = ['images/img3.jpg', 'images/img2.jpg', 'images/img1.png']
            img = cv2.imread(countdown_images[countdown])
            img1 = cv2.resize(img,(448, 448))
            
        img2 = cv2.resize(frame,(448, 448))
        #vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        #vis[:h1, :w1] = frame 
        #vis[:h2, w1:w1+w2] = res
        
        vis = np.concatenate((img1, img2), axis=1)
        cv2.imshow('frame: ',vis)

    def Play(self):
        print(self.opts)
        print(self.username)
        
        self.wins = 0;
        self.losses = 0;
        self.draws = 0;
        
        difficulty = 1
        if(self.username in self.difficultNames):
            difficult = 2
        elif(self.username in self.easyNames):
            difficulty = 0
        
        if(self.opts[1] == "Neural"):
            det = DLClassifier("weights/detector.ckpt")
        else:
            det = CVClassifier(invert = False)
         

        cap = cv2.VideoCapture(0)
        
        
        self.countDown(start=True)
        
        while(True):
            #må putte inn y i countDown
            count = self.countDown()
            ret, frame = cap.read()
            
            if(count >= 3):
                det.detect_from_cvmat(frame)
                oponent = self.selectHand(det.result,difficulty)
                if(det.result == oponent):
                    #Draw
                    self.draws +=1
                    result ="Draw"
                elif((det.result + 2)%3 == oponent):
                    #Won
                    self.wins += 1
                    result = "Win" 
                else:
                    #loose
                    self.losses += 1
                    result = "Loose"
                self.display(det.image,count,oponent,result)
            else:
                #Display countdown
                self.display(frame,count)
                
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        
        
        
        
if __name__ == "__main__":
    app = RPS_Game()
    app.run()
    pass





