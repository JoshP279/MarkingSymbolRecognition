from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
import numpy as np
import os
import cv2

def preprocess_and_save_images(input_folder, output_folder, invert, img_size=(48, 48), color_mode='grayscale'):
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

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

        if invert:
            img = cv2.flip(img, 1)  # Flip image horizontally to simulate left-handed ticks

        if color_mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

# Preprocess and save inverted images for each category to simulate left-handed ticks
invert = False
preprocess_and_save_images('raw_data/ticks', 'data/ticks', img_size=(48, 48), color_mode='grayscale', invert=invert)
preprocess_and_save_images('raw_data/half_ticks', 'data/half_ticks', img_size=(48, 48), color_mode='grayscale', invert=invert)
preprocess_and_save_images('raw_data/messy_half_ticks', 'data/messy_half_ticks', img_size=(48, 48), color_mode='grayscale', invert=invert)
preprocess_and_save_images('raw_data/messy_ticks', 'data/messy_ticks', img_size=(48, 48), color_mode='grayscale', invert=invert)


def load_images_from_folder(folder, label, img_size=(48, 48)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img[..., np.newaxis]
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data for left-handed ticks, half-ticks, and neither
ticks_images, ticks_labels = load_images_from_folder('data/ticks', 0)
half_ticks_images, half_ticks_labels = load_images_from_folder('data/half_ticks', 1)
messy_half_ticks_images, messy_half_ticks_labels = load_images_from_folder('data/messy_half_ticks', 2)
messy_ticks_images, messy_ticks_labels = load_images_from_folder('data/messy_ticks', 3)

# Combine data including messy ticks
X = np.concatenate([ticks_images, half_ticks_images, messy_half_ticks_images, messy_ticks_images], axis=0)
y = np.concatenate([ticks_labels, half_ticks_labels, messy_half_ticks_labels, messy_ticks_labels], axis=0)

X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
class_weight_dict = dict(zip(unique_classes, class_weights))

class_weight_dict[0] *= 3 # Regular ticks
class_weight_dict[1] *= 0.5  # Half-ticks
class_weight_dict[2] *= 1.0  # Messy half-ticks
class_weight_dict[3] *= 3  # Messy ticks



model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.3),  # Dropout after pooling layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),  # Dropout after pooling layer
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),  # Dropout after pooling layer
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Dropout before fully connected layer
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=30,            # Increase rotation range
    width_shift_range=0.2,        # Shift the image horizontally
    height_shift_range=0.2,       # Shift the image vertically
    shear_range=0.2,              # Shearing transformations
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Randomly flip images
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    fill_mode='nearest'           # Fill in pixels after transformation
)



early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          epochs=6, 
          validation_data=(X_val, y_val), 
          class_weight=class_weight_dict,
          callbacks=[early_stopping])


if (invert): 
    model.save('left_handed_ticks.h5')
else:
    model.save('right_handed_ticks.h5')
