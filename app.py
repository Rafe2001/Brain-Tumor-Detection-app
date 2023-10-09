from flask import Flask, render_template, request, redirect, url_for
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load your brain tumor classification model
model = load_model('BrainTumor_categorical.h5')

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict_tumor():
    try:
        # Check if a file was submitted with the request
        if 'image' not in request.files:
            return redirect(url_for('index', result='No image file provided'))
        
        file = request.files['image']
        
        # Check if the file is empty
        if file.filename == '':
            return redirect(url_for('index', result='No selected file'))
        
        # Read the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Resize the image to match the model's input size (64x64)
        img = Image.fromarray(image)
        img = img.resize((64, 64))
        img = np.array(img)
        
        # Prepare the image for prediction
        input_img = np.expand_dims(img, axis=0)
        
        # Make predictions
        predictions = model.predict(input_img)
        predicted_label = np.argmax(predictions, axis=1)[0]
        
        result = 'Tumor detected' if predicted_label == 1 else 'No tumor detected'
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', result=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
