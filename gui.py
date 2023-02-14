import sys
import argparse
import re
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk
from tkinter import Label
from tkinter import Button
from tkinter import Message
from tkinter import messagebox
from tkinter import ttk
from tkinter import font

class App(Tk):
    filepath = ""
    def __init__(self):
        super().__init__()

        # configure the root window
        self.title('My Awesome App')
        self.geometry('500x100')

        # label
        self.label = ttk.Label(self, text='Hello World, We are group 13!', font=('Times', 20))
        self.label.pack()

        # button
        myFont = font.Font(size=15)
        self.button = Button(self, text='Please upload your MP4 file', height=2, width=24, bg='#FFE15D')
        self.button['font'] = myFont
        self.button['command'] = self.button_clicked
        self.button.pack()

    def button_clicked(self):
        f_types = [('Mp4 Files', '*.mp4')]
        filename = filedialog.askopenfilename(filetypes=f_types)  
        self.filepath = filename 
        self.destroy()

def detect_classify_display(frame, model ,model_type='resnet'):
    emotions = ['angry', 'disgust', 'fear',
                'happy', 'sad', 'surprise', 'neutral']
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face = frame_gray[y:y+h, x:x+w]
        if model_type == 'resnet50':
            face = cv2.resize(face, (197, 197)) / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
            face = np.repeat(face, 3, -1)
        else:
            face = cv2.resize(face, (48, 48)) / 255
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)
        predictions = model.predict(face)
        pred = np.argmax(predictions)
        prob = predictions[0, pred] * 100
        frame = cv2.rectangle(
            frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
        cv2.putText(frame, emotions[pred] + f'   {prob:.2f}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.imshow('Emotion recognition', frame)
    return emotions[pred]
    
def guimain():
    app = App()
    app.mainloop()
    filename = app.filepath
    print(filename)
    filename = filename.replace("\\", "/")
    return filename
    
if __name__ == "__main__":
    
    filename = guimain() 
    print(filename)
    
    angry = 0
    disgust = 0
    fear = 0
    happy = 0
    sad = 0
    surprise = 0
    neutral = 0
    
    parser = argparse.ArgumentParser(description='Webcam - detect classify display.')
    parser.add_argument('--model_path', help='Path to model.', default='models/fer2013_simple_CNN_1-e50-a0.64.hdf5')
    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    
    # Loading model
    saved_models_path = "models/"
    model_name = "fer2013_simple_CNN_1-e50-a0.64.hdf5"
    model_path = args.model_path
    model_type_match = re.search(r'(?<=fer2013_)(.*?)(?=_)', model_name)
    model_type = model_type_match.group()
    # model_type = re.search(r'(?<=fer2013_)(.*?)(?=_)', model_name)
    model = load_model(saved_models_path + model_name)
    
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened:
        print('Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('No captured frame -- Break!')
            break
        result = detect_classify_display(frame, model ,model_type)
        if result=="angry": 
            angry+=1
        elif result=="disgust":
            disgust+=1
        elif result=="fear":
            fear+=1
        elif result=="happy":
            happy+=1
        elif result=="sad":
            sad+=1
        elif result=="surprise":
            surprise+=1
        else:
            neutral+=1
        if cv2.waitKey(10) == 27:
            break

    data = {'angry':angry, 'disgust':disgust, 'fear':fear, 'happy':happy, 'sad': sad, 'surprise': surprise,
            'neutral':neutral}   

    courses = list(data.keys())
    values = list(data.values())
    
    #draw bar chart
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(courses, values, color ='maroon',
            width = 0.4)
    
    plt.xlabel("Emotion")
    plt.ylabel("Number of this emotion")
    plt.title("Emotion of this people")
    
    # draw pie chart
    # Creating explode data
    explode = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Creating color parameters
    colors = ( "orange", "cyan", "brown",
                        "grey", "indigo", "beige", "pink")

    # Wedge properties
    wp = { 'linewidth' : 1, 'edgecolor' : "green" }

    # Creating autocpt arguments
    def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} times)".format(pct, absolute)
        
    # Creating plot
    fig, ax = plt.subplots(figsize =(7, 7))
    wedges, texts, autotexts = ax.pie(values,
                                        autopct = lambda pct: func(pct, values),
                                        explode = explode,
                                        labels = courses, 
                                        shadow = True,
                                        colors = colors,
                                        startangle = 90,
                                        wedgeprops = wp,
                                        textprops = dict(color ="magenta"))

    # Adding legend
    ax.legend(wedges, courses,
                        title ="Emotions",
                        loc ="center left",
                        bbox_to_anchor =(1, 0, 0.5, 1))

    plt.setp(autotexts, size = 8, weight ="bold")
    ax.set_title("Customizing pie chart")

    max_key = max(data, key=data.get)
    notice = "This person is " + max_key
    
    # root = Tk()
    # root.geometry("300x100")
        
    # notice = "This person is " + max_key
    # w = Label(root, text ='Result', font = "50") 
    # w.pack()
        
    # msg = Message( root, text = notice)  
    # msg.pack()  
        
    # root.mainloop() 
    
    messagebox.showinfo("Result", notice)

    # show plot
    plt.show()


  
