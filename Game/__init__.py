import tkinter as tk
from PIL import Image
from PIL import ImageTk
from scipy import signal
import numpy as np
import time
import random
import cv2

class RPS_Game():
    def __init__(self):
        # GUI variabels
        self.opts = ["",""]
        self.username = ""
        self.initMainMenu()
        # GAME Countdown variabels
        
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
            self.hand_time = []
            self.hand_pos = []
        
        # Use timer if no hand position is given
        if hand_position == None:
            return (int(round(time.time() * 1000)) - self.start_time)//1000 #return seconds since start
        else:
            self.hand_pos.append(hand_position)
            self.hand_time.append((int(round(time.time() * 1000)) - self.start_time)/1000 + 1)
            return signal.find_peaks_cwt(np.array(self.hand_pos), np.array(self.hand_time))
            
            
        
    def run(self):
        self.window.mainloop()

    def Play(self):
        print(self.opts)
        print(self.username)
        
        
if __name__ == "__main__":
    app = RPS_Game()
    app.run()
    pass





