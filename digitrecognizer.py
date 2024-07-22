import tkinter as tk
from tkinter import Label, Button, Frame
from tkinter import filedialog
from PIL import Image, ImageTk

import tensorflow as tf
import numpy as np
import cv2

#Function to upload image from file
def imageUpload():
    #Getting our neural network model
    model = tf.keras.models.load_model('DigitRecognizer/model/digitrecognizer.keras')

    
    fileTypes = [("Image files", "*.png *.jpg *.jpeg")]
    path = filedialog.askopenfilename(filetypes=fileTypes)

    if path:
        #Loading the image using open cv for predicting it
        img = cv2.imread(path)[:,:,0]
        img = cv2.resize(img, (28,28))
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("It probably is a {}".format(np.argmax(prediction)))

        #Loading and placing the image on the frame
        imgOut = Image.open(path)
        imgOut = imgOut.resize((75,75))
        pic = ImageTk.PhotoImage(imgOut)
        finalImgLabel.config(image=pic)
        finalImgLabel.image = pic

        finalText= "This probably is a " + str(np.argmax(prediction))

        predLabel.config(text=finalText)
    else:
        print("No file chosen")


#Creating the tkinter object and the main GUI frame
app = tk.Tk()
app.title("Digit Recognizer")
app.geometry("500x500")

#Logo and Title
main_frame = Frame(app)
main_frame.pack(expand=True)

img = Image.open('DigitRecognizer/images/crosshairs-solid.png')
img = img.resize((100,100))
img = ImageTk.PhotoImage(img)
imgLabel = Label(main_frame, image=img)
imgLabel.pack()

title = Label(main_frame, text="Digit Recognizer", font=("Arial", 25))
title.pack()

# Uploading Image button
uploadImage = Button(main_frame, text='Upload Image', activebackground="blue", activeforeground="white", padx=10, pady=4, cursor="hand2", command=imageUpload)
uploadImage.pack(pady=20)

#Final Image Showing Label
finalImgLabel = Label(main_frame)
finalImgLabel.pack(pady=20)

predLabel = Label(main_frame)
predLabel.pack(pady=10)

app.mainloop()