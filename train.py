import os
import cv2
import numpy as np
from tkinter import filedialog, messagebox
from tkinter import simpledialog


class Train:
    def __init__(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier('F:\Computer-Vision-with-Python\DATA\haarcascades\haarcascade_frontalface_default.xml')
        self.path = "F:\\faceimg"
        try:
            self.fun()
            print(np.array(self.faceSamples).shape)
            recognizer.train(np.array(self.faceSamples), np.array(self.id))
            recognizer.write('F://trainer//trainer3.yml')
        except:
            messagebox.showerror("Error", "File editable is not allowed")



    def fun(self):

        self.imagePaths = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        print(self.imagePaths)
        self.faceSamples = []
        self.id = []
        for imagePath in self.imagePaths:

            self.img = cv2.imread(imagePath, 0)
            self.img = cv2.resize(self.img, (200, 200))
            # cv2.imshow('frame',img)
            # cv2.waitKey(0)
            # cv2.destroyWindow('frame')
            #self.Id=imagePath.split('.')[1]
            self.Id = int(os.path.split(imagePath)[-1].split(".")[1])
            self.faces = self.detector.detectMultiScale(self.img)

            for (x, y, w, h) in self.faces:
                self.faceSamples.append(self.img[y:y + h, x:x + w])
                self.id.append(self.Id)
        print(np.array(self.faceSamples).shape)





