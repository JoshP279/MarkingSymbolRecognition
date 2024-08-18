import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load the EMNIST dataset
def load_emnist_subset():
    emnist = tf.keras.datasets.mnist.load_data(path="emnist-byclass.mat")
    (x_train, y_train), (_, _) = emnist
    
    # Dictionary to store selected samples from each class
    selected_samples = {}
    
    # Define the classes you want to include in the "neither" category (e.g., letters that resemble ticks)
    target_classes = [10, 11, 12, 13]  # Example: Classes corresponding to specific letters

    # Initialize selected samples for each target class
    for target_class in target_classes:
        selected_samples[target_class] = []
    
    # Iterate over the dataset and select a few samples from each target class
    for img, label in zip(x_train, y_train):
        if label in target_classes and len(selected_samples[label]) < 3:  # Choose 3 samples of each class
            selected_samples[label].append(img)
    
    # Combine the selected samples into arrays
    x_selected = []
    y_selected = []
    
    for label, images in selected_samples.items():
        for img in images:
            x_selected.append(img)
            y_selected.append(2)  # Assign a label of 2 for "neither"
    
    x_selected = np.array(x_selected)
    y_selected = np.array(y_selected)
    
    return x_selected, y_selected

# Preprocess EMNIST data
def preprocess_emnist_data(x_data, img_size=(48, 48)):
    if len(x_data) == 0:
        return np.array([])  # Return an empty array if no data
    x_data_resized = []
    for img in x_data:
        img_resized = cv2.resize(img, img_size)
        img_resized = img_resized / 255.0  # Normalize the image
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
        x_data_resized.append(img_resized)
    return np.array(x_data_resized)

# Load and preprocess a small subset of EMNIST data
x_emnist_subset, y_emnist_subset = load_emnist_subset()

# Check if EMNIST subset is empty
if len(x_emnist_subset) == 0:
    print("No EMNIST data selected. Skipping EMNIST data.")
else:
    # Reshape EMNIST data to match the dimensions of the custom data
    if len(x_emnist_subset.shape) == 3:  # If it's (samples, height, width), add a channel dimension
        x_emnist_subset = np.expand_dims(x_emnist_subset, axis=-1)

    # Preprocess EMNIST data
    x_emnist_subset = preprocess_emnist_data(x_emnist_subset)

# Load your custom tick and half-tick images
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

ticks_images, ticks_labels = load_images_from_folder('data/ticks', 0)
half_ticks_images, half_ticks_labels = load_images_from_folder('data/half_ticks', 1)

# Ensure dimensions match by adding a channel dimension to your custom data
ticks_images = ticks_images[..., np.newaxis]  # Add channel dimension
half_ticks_images = half_ticks_images[..., np.newaxis]  # Add channel dimension

# Combine custom tick and half-tick data
x_custom = np.concatenate((ticks_images, half_ticks_images), axis=0)
y_custom = np.concatenate((ticks_labels, half_ticks_labels), axis=0)

# Normalize the custom data (EMNIST is already normalized in preprocess_emnist_data)
x_custom = x_custom / 255.0

# Combine the data if EMNIST is not empty
if len(x_emnist_subset) > 0:
    x_combined = np.concatenate((x_emnist_subset, x_custom), axis=0)
    y_combined = np.concatenate((y_emnist_subset, y_custom), axis=0)
else:
    x_combined = x_custom
    y_combined = y_custom

# One-hot encode labels for multi-class classification
y_combined = to_categorical(y_combined, num_classes=3)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
y_train_classes = np.argmax(y_train, axis=1)  # Convert one-hot to class indices
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_classes), y=y_train_classes)
class_weights_dict = dict(enumerate(class_weights))

# Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: tick, half tick, neither
])

# Compile and Train the Model with increased epochs
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=30, validation_data=(X_val, y_val), class_weight=class_weights_dict)

# Save the Model
model.save('tick_detection_model.h5')
