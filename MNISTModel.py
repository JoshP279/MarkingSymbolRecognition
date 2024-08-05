import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D
from SymbolRecognitionModel import SymbolRecognitionModel

class MNISTModel:
    def __init__(self):
        self.IMG_SIZE = 28
        self.model = None
        self.class_samples = {}
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
        for class_idx in range(10):
            class_sample_idx = np.where(y_train == class_idx)[0][0]
            self.class_samples[class_idx] = x_train[class_sample_idx]

    def build_model(self):
        self.model = Sequential([
            Conv2D(64, (3, 3), input_shape=self.x_trainr.shape[1:], activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.summary()

    def train_model(self, epochs=3, validation_split=0.3):
        if self.model is None:
            self.build_model()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_trainr, self.y_train, epochs=epochs, validation_split=validation_split)

    def save_model(self, model_path="mnist_model.h5"):
        self.model.save(model_path) if self.model else print("No model to save")

    def load_model(self, model_path="mnist_model.h5"):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def predict_custom_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Image at path {image_path} could not be loaded.")
            return
        img = self.preprocess_image(img)
        img = img.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f"The result is probably: {predicted_class} with confidence: {confidence:.2f}")

    def preprocess_image(self, img):
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # resize and pad image code goes here
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        return img / 255.0

    def convert_pdf_to_images(self, pdf_path):
        document = fitz.open(pdf_path)
        images = []
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append(img_data)
        return images
    
    def predict_from_pdf(self, pdf_path):
        images = self.convert_pdf_to_images(pdf_path)
        for img_data in images:
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Could not decode image from PDF.")
                continue
            predictions = self.segment_and_predict(img)
            for predicted_class, confidence, (x, y, w, h) in predictions:
                print(f"Predicted: {predicted_class} with confidence: {confidence:.2f} at ({x}, {y}, {w}, {h})")

    def segment_and_predict(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        predictions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 1.3 and 100 < cv2.contourArea(cnt) < 8000:
                digit = img[y:y+h, x:x+w]
                digit = self.resize_and_pad(digit, self.IMG_SIZE)
                digit_normalized = digit.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 1) / 255.0
                pred = self.model.predict(digit_normalized)
                predicted_class = np.argmax(pred)
                confidence = np.max(pred)
                if confidence > 0.9:
                    predictions.append((predicted_class, confidence, (x, y, w, h)))
                    plt.figure(figsize=(2, 2))
                    plt.imshow(digit, cmap='gray')
                    plt.title(f'Predicted: {predicted_class}, Conf: {confidence:.2f}')
                    plt.axis('off')
                    plt.show()
                else:
                    plt.imshow(digit, cmap='gray')
                    plt.title(f'Low Confidence Predicted: {predicted_class}, Conf: {confidence:.2f}')
                    plt.axis('off')
                    plt.show()
        return predictions



    def resize_and_pad(self, img, size):
        h, w = img.shape
        if h > w:
            new_h, new_w = size, int(w * (size / h))
        else:
            new_h, new_w = int(h * (size / w)), size

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2

        padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if padded.shape[0] < size or padded.shape[1] < size:
            padded = cv2.copyMakeBorder(padded, 0, size - padded.shape[0], 0, size - padded.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded