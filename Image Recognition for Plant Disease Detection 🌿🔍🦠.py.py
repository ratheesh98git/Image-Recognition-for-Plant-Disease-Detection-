import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

app = Flask(__name__)

model = load_model('plant_disease_model.h5')

img_size = (256, 256)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0 
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0) 
    prediction = model.predict(img)
    disease_class = np.argmax(prediction, axis=1)[0]
    return disease_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)
    disease_class = predict_disease(image_path)
    response = {
        'class': disease_class
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
