import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request,render_template,redirect,url_for
import os
from werkzeug.utils import secure_filename
import openpyxl
app = Flask(__name__)

model = load_model('fruitModel')
model1 = load_model('VegModel')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(128, 128))
    show_img = image.load_img(img_path, grayscale=False, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    
    return preds



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/care')
def register():
    return render_template('care.html')

@app.route('/remedy')
def remedy():
    return render_template('remedy.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        t = request.form['txt']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        Vegdf = pd.read_excel('Veg.xlsx')
        Fruitdf = pd.read_excel('Fruit.xlsx')
        if t == 'fruit':

            preds = model_predict(file_path, model)
            print(preds[0])

            # x = x.reshape([64, 64]);
            disease_class = ['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___healthy',
                            'Corn_(maize)___Northern_Leaf_Blight', 'Peach___Bacterial_spot', 'Peach___healthy']
            a = preds[0]
            ind=np.argmax(a)
            print('Prediction:', disease_class[ind])
            print(Fruitdf.iloc[[ind]]['caution'])
            result=disease_class[ind]

        else:
            preds = model_predict(file_path, model1)
            print(preds[0])
            # x = x.reshape([64, 64]);
            disease_class = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                            'Potato___healthy', 'Potato___Late_blight', 'Tomato___Bacterial_spot','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot']
            a = preds[0]
            ind=np.argmax(a)
            print('Prediction:', disease_class[ind])
            print(Vegdf.iloc[[ind]]['caution'])
            result=disease_class[ind]
        return result
    return None

if __name__ == "main":
    app.run(debug=True)