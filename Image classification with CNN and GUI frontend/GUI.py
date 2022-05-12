from tkinter import filedialog
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from keras.models import load_model
import numpy as np
import math
#load the trained model to classify traffic signs
model = load_model('image_classifier.h5')
#dictionary to label all object class.
str_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
classes = { 1:'Airplane',
            2:'Automobile',
            3:'Bird',
            4:'Cat',
            5:'Deer',
            6:'Dog',
            7:'Frog',
            8:'Horse',
            9:'Ship',
            10:'Truck'}
#initialize GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Basic Image Classifier')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    print(pred)
    index = np.where(pred >= 0)
    print("index")
    print(index)
    sign = " "
    for x in index[0]:
        sign += "{Class} with %{percent} certainty \n".format(Class = classes[x+1], percent = math.floor(pred[x]/10))
    print(sign)
    label.configure(foreground='#011638', text=sign)
def show_classify_button(file_path):
    classify_b=Button(top,text="Click to Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Upload an image from file",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Upload an image of an airplane, automobile, bird, cat, deer, \n dog, frog, horse, ship or truck",pady=25, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()