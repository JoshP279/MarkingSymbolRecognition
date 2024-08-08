import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Step 1: Preprocess and Save the Images
def preprocess_and_save_images(input_folder, output_folder, img_size=(48, 48), color_mode='grayscale'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        if color_mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, img_size)

        # If grayscale, convert to 3-channel RGB for consistency in input shape
        if color_mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

# Preprocess and save images for each category
preprocess_and_save_images('raw_data/ticks', 'data/ticks', img_size=(48, 48), color_mode='grayscale')
preprocess_and_save_images('raw_data/others', 'data/others', img_size=(48, 48), color_mode='grayscale')

# Step 2: Load the Preprocessed Data
def load_images_from_folder(folder, label, img_size=(48, 48)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data from preprocessed folders
ticks_images, ticks_labels = load_images_from_folder('data/ticks', 1)
other_images, other_labels = load_images_from_folder('data/others', 0)

# Combine and shuffle data
X = np.concatenate((ticks_images, other_images), axis=0)
y = np.concatenate((ticks_labels, other_labels), axis=0)

# Normalize the data
X = X / 255.0
X = X.reshape(-1, 48, 48, 1)  # Add channel dimension

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: tick or no tick
])

# Step 4: Compile and Train the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Train the model using the training data generator and validation data
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Step 5: Save the Model
model.save('tick_detection_model.h5')
