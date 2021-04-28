import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog
import tkinter.ttk as ttk
import xlrd
import sys
from xlutils.copy import copy
import time
import openpyxl
def fun(root):
        Id=simpledialog.askstring("Id","Enter your Id",parent=root)
        if(Id==None):
            messagebox.showerror("Incorrect","Please Enter Your Id")
            return
        print(Id)
        id=str(Id)
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('F:\Computer-Vision-with-Python\DATA\haarcascades\haarcascade_frontalface_default.xml')


        sampleNum = 0
        while (True):
            ret, img = cam.read()
            faces = detector.detectMultiScale(img, 1.4, 6)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w+10, y + h+10), (255, 0, 0), 2)

                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder
                cv2.imwrite("F://faceimg//User." + id + '.' + str(sampleNum) + ".jpg", img[y:y + h, x:x + w])

                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(1) & 0xFF == 27:
                break
            # break if the sample number is morethan 20
            elif sampleNum > 30:
                break
            fps = int(cam.get(5))
            print("fps:", fps)
        cam.release()
        cv2.destroyAllWindows()
        try:
                rb = xlrd.open_workbook('C:\\Users\\acer\\Untitled Folder\\xlwt example.xls', formatting_info=True)
                r_sheet = rb.sheet_by_index(0)
                r_sheet.defcolwidth = 30
                nrows = r_sheet.nrows
                ncols = r_sheet.ncols
                print(ncols)
                print(nrows)
                wb = copy(rb)
                w_sheet = wb.get_sheet(0)
                w_sheet.write(nrows,0,Id)
                wb.save('C:\\Users\\acer\\Untitled Folder\\xlwt example.xls')
        except:
            messagebox.showerror("Window Close","close xls sheet ")
class Demo1:
    def __init__(self,master):
        self.master=master
        '''This class configures and populates the toplevel window.
                   top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.map('.', background=
        [('selected', _compcolor), ('active', _ana2color)])
        self.master.geometry("849x641+451+111")
        self.master.title("New Toplevel")
        self.master.configure(borderwidth="3")
        self.master.configure(background="#b5335e")
        self.master.configure(highlightcolor="#151d63")
        self.frame=tk.Frame(self.master)
        self.quitButton = tk.Button(self.frame, text='Quit', width=25, command=self.close_windows)
        self.quitButton.pack()
        self.frame.pack()

    def close_windows(self):
        self.master.destroy()

    def show(self,root):
        root.update()
        root.deiconify()

if __name__=='__main__':
    fun()