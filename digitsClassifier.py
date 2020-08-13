from PIL import Image, ImageDraw
import PIL
from tkinter import *
import tkinter as tk
import cv2
from keras.models import load_model
import numpy as np
import pandas

canvas_height = 450
canvas_width = 450

model = load_model('kerasCNNmodel.h5')

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry('800x690+400+100')
        self.config(bg='white')
        self.resizable(0, 0)
        self.x = self.y = 0
        self.image = PIL.Image.new("RGB", (canvas_width, canvas_height))

        # Creating elements
        self.head = Label(self, text="ML HandwrittenDigits Classifier", bg='white', font=("Helvetica", 24), width=22)
        self.frame1 = Frame(self, bg='white')
        self.canvas = Canvas(self.frame1, width=canvas_width, height=canvas_height, bg='white', relief=SUNKEN, bd=8, cursor='cross')
        self.label = Label(self.frame1, text="Thinking..", font=("Helvetica", 36), bg='white')
        self.frame2 = Frame(self, bg='white')
        self.classify_btn = Button(self.frame2, text="Classify", font=("Helvetica", 22), width=15, command=self.classify_handwriting)
        self.button_clear = Button(self.frame2, text="Clear", font=("Helvetica", 22), width=15, command=self.clear_all)

        # Pack
        self.head.pack(side=TOP, fill=BOTH, pady=20)
        self.frame1.pack(side=TOP, pady=10, padx=5)
        self.canvas.pack(side=LEFT, padx=20)
        self.label.pack(side=LEFT, pady=2, padx=2)
        self.frame2.pack(side=TOP, pady=20)
        self.classify_btn.pack(side=LEFT, pady=2, padx=10)
        self.button_clear.pack(side=LEFT, pady=2, padx=10)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("RGB", (canvas_width, canvas_height))

    def classify_handwriting(self):
        im = cv2.imread('image.png', 0)
        im = cv2.resize(im, (28, 28))
        im = np.expand_dims(im, axis=2)
        im = np.expand_dims(im, axis=0)
        img = im / 255.0
        res = model.predict(img)
        a = np.argmax(res)
        b = max(max(res))
        print(a)
        print(b)
        self.label.configure(text=str(a) + ', ' + str(int(b * 100)) + '%')
        # cv2.imshow('sas', im)
        # cv2.waitKey()

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 15
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
        draw = ImageDraw.Draw(self.image)
        draw.ellipse([self.x - r, self.y - r, self.x + r, self.y + r], fill='white')
        self.image.save('image.png')


app = App()
mainloop()




