from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
import numpy as np
import os
import cv2

def preprocess_and_save_images(input_folder, output_folder, invert, img_size=(48, 48), color_mode='grayscale'):
    print(f'Preprocessing images from {input_folder} and saving to {output_folder}...')

    # Skip deletion for the 'non_ticks' folder
    if 'non_ticks' in output_folder:
        print("Skipping deletion of non-tick data.")
    else:
        if os.path.exists(output_folder):
            for file in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove existing files
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process and save images
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


def load_images_from_folder(folder, label, img_size=(48, 48)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize to 48x48
            img = img[..., np.newaxis]  # Add a new axis to make it (48, 48, 1) for grayscale
            images.append(img)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def filter_valid_data(images, labels, valid_range=(0, 4)):
    """Filters out data that contains labels outside the valid range."""
    valid_indices = np.where((labels >= valid_range[0]) & (labels <= valid_range[1]))[0]
    return images[valid_indices], labels[valid_indices]

# Preprocess and save inverted images for each category to simulate left-handed ticks
inverted = False

# preprocess_and_save_images('raw_data/ticks', 'data/ticks', img_size=(48, 48), color_mode='grayscale', invert=inverted)
# preprocess_and_save_images('raw_data/half_ticks', 'data/half_ticks', img_size=(48, 48), color_mode='grayscale', invert=inverted)
# preprocess_and_save_images('raw_data/messy_half_ticks', 'data/messy_half_ticks', img_size=(48, 48), color_mode='grayscale', invert=inverted)
# preprocess_and_save_images('raw_data/messy_ticks', 'data/messy_ticks', img_size=(48, 48), color_mode='grayscale', invert=inverted)
# preprocess_and_save_images('data/non_ticks', 'data/non_ticks', img_size=(48, 48), color_mode='grayscale', invert=inverted)  # Preprocess non-ticks

# Load data for ticks, half-ticks, messy ticks, messy half-ticks, and non-ticks
ticks_images, ticks_labels = load_images_from_folder('data/ticks', 0)
messy_ticks_images, messy_ticks_labels = load_images_from_folder('data/messy_ticks', 1)
half_ticks_images, half_ticks_labels = load_images_from_folder('data/half_ticks', 2)
messy_half_ticks_images, messy_half_ticks_labels = load_images_from_folder('data/messy_half_ticks', 3)
non_ticks_images, non_ticks_labels = load_images_from_folder('data/non_ticks', 4)  # Non-ticks label is 4
# Check shapes of the arrays
print(ticks_images.shape)  # Should be (num_samples, 48, 48, 1)
print(messy_ticks_images.shape)  # Same
print(half_ticks_images.shape)  # Same
print(messy_half_ticks_images.shape)  # Same
print(non_ticks_images.shape)  # Same

# Combine all data
X = np.concatenate([ticks_images, messy_ticks_images, half_ticks_images, messy_half_ticks_images, non_ticks_images], axis=0)
y = np.concatenate([ticks_labels, messy_ticks_labels, half_ticks_labels, messy_half_ticks_labels, non_ticks_labels], axis=0)

# Normalize images
X = X / 255.0

# Filter out problematic data (labels outside the range of 0 to 4)
X_filtered, y_filtered = filter_valid_data(X, y, valid_range=(0, 4))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

# Compute class weights to balance the dataset
unique_classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
class_weight_dict = dict(zip(unique_classes, class_weights))

class_weight_dict[0] *= 6  # Decrease the weight of ticks
class_weight_dict[1] *= 6  # Decrease the weight of messy ticks
class_weight_dict[2] *= 4  # Decrease the weight of half-ticks
class_weight_dict[3] *= 4  # Decrease the weight of messy half-ticks
class_weight_dict[4] *= 0.02  # Decrease the weight of non-ticks

# Define the CNN model
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
    Dense(5, activation='softmax')  # Update output to 5 classes (including non-tick class)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, 
          epochs=5, 
          batch_size=32, 
          validation_data=(X_val, y_val), 
          class_weight=class_weight_dict,
          callbacks=[early_stopping])

# Save the trained model based on inversion
model.save('left_handed_ticks.h5' if inverted else 'right_handed_ticks.h5')
