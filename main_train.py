import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
directory = 'brain_tumor_dataset/'

no_tumor = os.listdir(directory + 'no/')
yes_tumor = os.listdir(directory + 'yes/')

dataset = []
label = []
INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(directory + 'no/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(directory + 'yes/' + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize pixel values to [0, 1]
#X_train = normalize(X_train,axis=1)
#X_test = normalize(X_test, axis=1)

X_train = X_train / 255.0
X_test = X_test / 255.0


y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model with data augmentation
model.fit(X_train,y_train, 
          batch_size=32, 
          verbose=1, 
          epochs=20, 
          validation_data=(X_test,y_test), 
          shuffle=False
)

# Save the model
model.save('BrainTumor_categorical.h5')
