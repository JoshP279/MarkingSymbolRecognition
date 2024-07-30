import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D
import cv2
import matplotlib.pyplot as plt

class MNISTModel:
    def __init__(self):
        self.IMG_SIZE = 28
        self.model = None
        self.class_samples = {}  # To store one sample image per class
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = tf.keras.utils.normalize(x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(x_test, axis=1)
        self.y_train = y_train
        self.y_test = y_test
        self.x_trainr = np.array(self.x_train).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        self.x_testr = np.array(self.x_test).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
         # Store one sample image per class
        for class_idx in range(10):
            class_sample_idx = np.where(y_train == class_idx)[0][0]
            self.class_samples[class_idx] = x_train[class_sample_idx]


    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), input_shape=self.x_trainr.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))

        self.model.add(Dense(32))
        self.model.add(Activation('relu'))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.summary()

    def train_model(self, epochs=7, validation_split=0.3):
        if self.model is None:
            self.build_model()

        print("Total Training Samples: ", len(self.x_trainr))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_trainr, self.y_train, epochs=epochs, validation_split=validation_split)

    def save_model(self, model_path="mnist_model.h5"):
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save")

    def load_model(self, model_path="mnist_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def predict(self):
        if self.model:
            predictions = self.model.predict(self.x_testr)
            print(predictions)
        else:
            print("No model to predict with")

    def predict_custom_image(self, image_path):
        if not self.model:
            print("Model not loaded. Load or train a model first.")
            return

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Image at path {image_path} could not be loaded.")
            return

        original_img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = np.invert(np.array([original_img]))
        img = img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        print(f"The result is probably: {predicted_class} with confidence: {confidence:.2f}")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(self.class_samples[predicted_class], cmap=plt.cm.binary)
        axs[0].set_title("Original Image")
        axs[1].imshow(img[0], cmap=plt.cm.binary)
        axs[1].set_title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
        plt.show()
