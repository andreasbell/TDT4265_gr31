
# coding: utf-8

# In[61]:

import tkinter as tk
from PIL import Image
from PIL import ImageTk
import cv2

class RPS_Game():
    def __init__(self):
        self.opts = ["",""]
        self.username = ""
        self.initMainMenu()  
        
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
        
    def run(self):
        self.window.mainloop()

    def Play(self):
        print(self.opts)
        print(self.username)
        
    
app = RPS_Game()
app.run()





