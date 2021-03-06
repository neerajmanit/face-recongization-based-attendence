#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.19
#  in conjunction with Tcl version 8.6
#    Dec 24, 2019 11:58:59 PM IST  platform: Windows NT

import sys
import os
from tkinter import messagebox

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import unknown_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Toplevel1 (root)
    unknown_support.init(root, top)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = Toplevel1 (w)
    unknown_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#ececec' # Closest X11 color: 'gray92' 
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])
        self.master=top
        top.geometry("849x641+451+111")
        top.title("New Toplevel")
        top.configure(background="#bcbfd8")

        self.Label1 = tk.Label(top)
        self.Label1.place(relx=1.037, rely=0.461, height=756, width=1062)
        self.Label1.configure(background="#b6a0d8")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(highlightcolor="#646464")
        self._img1 = tk.PhotoImage(file="C:/Users/acer/Downloads/resize-1577071242113308523111a9922b266a38dea646a0da2ef38550.png")
        self.Label1.configure(image=self._img1)
        self.Label1.configure(text='''Label''')
        self.Label1.configure(width=1062)

        self.TProgressbar1 = ttk.Progressbar(top)
        self.TProgressbar1.place(relx=0.0, rely=0.948, relwidth=0.999
                , relheight=0.0, height=40)
        self.TProgressbar1.configure(length="1060")

        self.Button1 = tk.Button(top)
        self.Button1.place(relx=-0.132, rely=0.112, height=33, width=56)
        self.Button1.configure(activebackground="#ececec")
        self.Button1.configure(activeforeground="#000000")
        self.Button1.configure(background="#d9d9d9")
        self.Button1.configure(disabledforeground="#a3a3a3")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="#d9d9d9")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Button''')

        self.menubar = tk.Menu(top,font="TkMenuFont",bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)

        self.Button2 = tk.Button(top)
        self.Button2.place(relx=0.0, rely=0.0, height=763, width=1066)
        self.Button2.configure(activebackground="#ececec")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="#d9d9d9")
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="#d9d9d9")
        self.Button2.configure(highlightcolor="black")
        self._img2 = tk.PhotoImage(file="C:/Users/acer/Downloads/resize-1577071242113308523111a9922b266a38dea646a0da2ef38550.png")
        self.Button2.configure(image=self._img2)
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''Button''')
        self.Button2.configure(width=1066)
        self.Button2.configure(command=self.fun2)


    def fun2(self):
        import time
        self.TProgressbar1['value'] = 20
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 40
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 60
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 80
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 90
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 94
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 99
        root.update_idletasks()
        time.sleep(1)
        self.TProgressbar1['value'] = 100
        root.update_idletasks()
        ans=messagebox.askokcancel("access","Allow system to acces your media ")
        print(ans)
        if(ans):

            self.newWindow = tk.Toplevel(self.master)
            self.app = Demo2(self.newWindow)
        #os.system("C:\\Users\\acer\\Desktop\\gui2.py")



class Demo2:
    def __init__(self, master):
        self.master = master
        self.master.geometry("849x641+451+111")
        self.master.title("New Toplevel")
        self.master.configure(borderwidth="3")
        self.master.configure(background="#b5335e")
        self.master.configure(highlightcolor="#151d63")
        self.frame = tk.Frame(self.master)
        self.quitButton = tk.Button(self.frame, text = 'Quit', width = 25, command = self.close_windows)
        self.quitButton.pack()
        self.frame.pack()



    def close_windows(self):
        self.master.destroy()
if __name__ == '__main__':
    vp_start_gui()






