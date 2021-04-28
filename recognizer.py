import cv2
#import numpy as np
import xlrd
from xlutils.copy import copy
#import openpyxl
import datetime
from tkinter import messagebox
class Recognizer:
    def __init__(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('F://trainer//trainer3.yml')
        faceCascade = cv2.CascadeClassifier(
            'F:\Computer-Vision-with-Python\DATA\haarcascades\haarcascade_frontalface_default.xml')

        cam = cv2.VideoCapture(0)
        dict={}
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                print(Id,conf)

                if (conf < 100):
                    dict[Id] = 1
                    if (Id == 67):
                        Id = "ganesh"



                else:
                    Id = "Unknown"

                cv2.putText(img=im, text=str(Id), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(0, 255, 0),
                        thickness=2, lineType=cv2.LINE_AA)
            cv2.imshow('im', im)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        try:
                rb = xlrd.open_workbook('C:\\Users\\acer\\Untitled Folder\\xlwt example.xls',formatting_info=True)
                r_sheet = rb.sheet_by_index(0)
                r_sheet.defcolwidth=30
                nrows = r_sheet.nrows
                ncols = r_sheet.ncols
                today=str(datetime.datetime.now())
                print(ncols)
                print(nrows)
                wb = copy(rb)
                w_sheet = wb.get_sheet(0)
                w_sheet.write(0,ncols,today)
                for id in range(1,nrows):
                    print(r_sheet.cell_value(id,0))
                    if(dict.get(r_sheet.cell_value(id,0),0)):
                        w_sheet.write(id,ncols,1)
                    else:
                        w_sheet.write(id,ncols,0)
                wb.save('C:\\Users\\acer\\Untitled Folder\\xlwt example.xls')
        except:
                 messagebox('Close Window','Close xls sheet')