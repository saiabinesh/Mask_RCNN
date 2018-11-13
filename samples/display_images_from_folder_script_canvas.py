from tkinter import *
from PIL import Image, ImageTk
import os,time
#from Tkinter import *


if __name__ == "__main__":
    root = Tk()
def quit():
    global root
    root.quit()   
    
frame = Frame(root, bd=2, relief = SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

canvas = Canvas(frame, bd=1, width=1152, height=1152)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
frame.pack(fill=BOTH,expand=1)

ROOT_DIR = os.path.abspath("../")
#Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
file_names = os.listdir(IMAGE_DIR)
img = ImageTk.PhotoImage(Image.open(os.path.join(IMAGE_DIR, file_names[0])))
cimg = canvas.create_image(0,0,image=img,anchor="nw")
canvas.config(scrollregion=canvas.bbox(ALL))


    
    
def display_image(path):
    global canvas, img, FileDir
    #FileDir = os.path.join(IMAGE_DIR, File[f])
    #canvas.destroy()
    #canvas.delete("all")
    #del img
    #del canvas
    #canvas = Canvas(frame, bd=1, width=950, height=600)
    #canvas.grid(row=0, column=0, sticky=N+S+E+W)
    img = ImageTk.PhotoImage(Image.open(path))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))
    #root.after(10000,display_image)
    #root.update_idletasks()
    #root.after(1000,display_image)
    #root.destroy()
    #root.mainloop()

while True:
    #Button(root, text="Quit", command=quit).pack()
    root.mainloop()
    #do something
    #Load all files from directory
    #File = os.listdir(IMAGE_DIR)
    #print(File)



    #setting up a tkinter frame and canvas
    



    for image_name in file_names:    
        #total = sum(range(100000000))
        #print(total)
        print(image_name)
        display_image((os.path.join(IMAGE_DIR, image_name)))
        time.sleep (3)
        
      

        