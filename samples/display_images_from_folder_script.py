import os, time
import tkinter as tk
from PIL import ImageTk, Image

def display_image(path):
    print("Displaying image: ",path)
    
    #This creates the main window of an application
    window = tk.Toplevel()
    window.title("Join")
    window.geometry("1152x1152")
    window.configure(background='grey')
    
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(path))

    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image = img)

    #The Pack geometry manager packs widgets in rows or columns.
    panel.pack(side = "bottom", fill = "both", expand = "yes")

    
    
ROOT_DIR = os.path.abspath("../")
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

file_names = os.listdir(IMAGE_DIR)
for image_name in file_names:
    #time.sleep (1)
    total = sum(range(100000000))
    print(total)
    print(image_name)
    display_image((os.path.join(IMAGE_DIR, image_name)))
    
#Start the GUI
window.mainloop()    