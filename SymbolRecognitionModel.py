from keras.models import load_model
import cv2
import numpy as np
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import os

class SymbolRecognitionModel:
    def __init__(self, model_path):
        # Load the Keras model
        self.model = load_model(model_path)
        self.IMG_SIZE = 48  # Model expects 48x48 input images

    def preprocess_image(self, img):
        """ Resize and normalize the image for the model. """
        if len(img.shape) == 2:  # Image is already grayscale
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Image is in color
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unexpected image shape, cannot preprocess image.")

        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img


    def predict_symbol(self, img):
        """ Predict if the image contains a tick or a cross. """
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_class, confidence  # Return the class and the confidence of the prediction

    def convert_pdf_to_images(self, pdf_path):
        document = fitz.open(pdf_path)
        images = []
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append(img_data)
        return images

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
                symbol = img[y:y+h, x:x+w]
                symbol = self.resize_and_pad(symbol, self.IMG_SIZE)
                predicted_class, confidence = self.predict_symbol(symbol)
                print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")
                if confidence > 0.3:
                    predictions.append((predicted_class, confidence, (x, y, w, h)))
                    plt.figure(figsize=(2, 2))
                    plt.imshow(symbol, cmap='gray')
                    plt.title(f'Predicted: {predicted_class}, Conf: {confidence:.2f}')
                    plt.axis('off')
                    plt.show()
                else:
                    plt.figure(figsize=(2, 2))
                    plt.imshow(symbol, cmap='gray')
                    plt.title(f'Low Conf Predicted: {predicted_class}, Conf: {confidence:.2f}')
                    plt.axis('off')
                    plt.show()
        return predictions

    def predict_from_pdf(self, pdf_path):
        ticks = 0
        images = self.convert_pdf_to_images(pdf_path)
        for img_data in images:
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("Could not decode image from PDF.")
                continue
            predictions = self.segment_and_predict(img)
            for predicted_class, confidence, (x, y, w, h) in predictions:
                if predicted_class == 0:
                    ticks+=1
                    print(f"Tick detected with confidence {confidence:.2f} at position ({x}, {y}, {w}, {h})")
                else:
                    print(f"Other symbol detected with confidence {confidence:.2f} at position ({x}, {y}, {w}, {h})")

        print(f"Total ticks detected: {ticks}")
