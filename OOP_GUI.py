from tkinter import Tk, Canvas, Button, Label
from tkinter import filedialog as fd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image, ImageTk



class my_Programm:
    def __init__(self, master):
        self.master = master
        master.title = 'Classificator'

        self.model = load_model('model_6.h5')

        self.classification_labels = {0: 'Автокран',
          1: 'Легковой автомобиль',
          2: 'Экскаватор',
          3: 'Человек',
          4: 'Самосвал',
          5: 'Карьерный погрузчик',
          6: 'Каток',
          7: 'Бульдозер'}
        
        self.label = Label(master, text='First Label')
        self.label.pack()

        self.file_choose_button = Button(master, text='Image',
                                         command=self.test_image)
        self.file_choose_button.pack()

        #self.canvas = Canvas(master, height=400, width=400)
        #self.canvas.pack()

        #self.image = Image.open(fn)
        self.image = None

        
        self.image_label = Label(master)   
        #self.image_label = Label(master, image=self.photo_image)
        self.image_label.pack()

        self.result_label = Label(master, text='what is it?')
        self.result_label.pack()

    def test_image(self):
        file_name = fd.askopenfilename()
        self.display_image(file_name)

    def display_image(self, fn):
        self.image = Image.open(fn)
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.image_label.configure(image=self.photo_image)
        #self.image_label.image=self.photo_image # нафига еще и эта строчка

        self.my_pred(fn)

    def my_pred(self, f):
    
        img = image.load_img(f, target_size=(160, 160))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = expanded_img_array / 255.  # Preprocess the image
        prediction = self.model.predict(preprocessed_img)
        prediction = prediction.tolist()
        a = prediction[0].index(max(prediction[0]))
        r = self.classification_labels[a]
        self.result_label.configure(text=r)
        #print(np.where(prediction==max(prediction)))
        #print(validation_generator.class_indices)

root = Tk()
my_Programm(root)
root.mainloop()
