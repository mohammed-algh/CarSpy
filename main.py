from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
import cv2
import os
import numpy as np
from preprocessing import *

img_width = 448
img_height = 448
data = load_csv(nrows=100).values
imgs = []
targets = []
for x in data:
    # Load image using OpenCV
    img = cv2.imread(f'car_ims/cars_train/{x[0]}')
    # Resize image
    resized_img = cv2.resize(img, (img_width, img_height))
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # Add alpha channel with all values set to 255 (fully opaque)
    rgba_img = np.dstack((rgb_img, np.full((448, 448), 255)))
    # Convert to tensor
    rgba_tensor = tf.convert_to_tensor(rgba_img, dtype=tf.float32)
    imgs.append(rgba_tensor)
    targets.append([x[1], x[2], x[3], x[4]])

# data_prepared = prepare(data).values[:, -1]
# target = extract_target(data)

# # print(type(features))
# # print(type(features[0][0]))
# # print(type(features[0][0]))
# # Define the dimensions of the input images


# # Define the number of classes (in this case, objects) to be detected
num_classes = 4


# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(img_height, img_width, 4)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flattening the output of previous layer
model.add(Flatten())

# Output layers: two dense layers for bounding box coordinates and one dense layer for class label prediction
model.add(Dense(256, activation='relu'))
# Predict 4 coordinates for bounding box
model.add(Dense(8, activation='linear'))

# Predict the class label for object
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(np.array(imgs), np.array(targets), epochs=18, batch_size=10)

model.save("ahalop.h5")

img = cv2.imread(f'car_ims/cars_train/00205.jpg')
resized_img = cv2.resize(img, (448, 448))
rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
rgba_img = np.dstack((rgb_img, np.full((448, 448), 255)))
rgba_img_expanded = np.expand_dims(rgba_img, axis=0)

prediction = model.predict(rgba_img_expanded)
print(prediction)
