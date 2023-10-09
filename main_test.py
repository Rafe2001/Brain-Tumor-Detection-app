import cv2
import os
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('BrainTumor_categorical.h5')

image = cv2.imread('Prediction\pred1.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))  # Assign the resized image back to 'img'

img = np.array(img)

input_img = np.expand_dims(img, axis=0)
predictions = model.predict([input_img])
predicted_label = np.argmax(predictions, axis=1)[0]

print(f"Predicted class label: {predicted_label}")
