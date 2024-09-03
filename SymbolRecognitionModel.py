from keras.models import load_model
import cv2
import numpy as np
import fitz  # pip install PyMuPDF
import matplotlib.pyplot as plt
import os

class SymbolRecognitionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.IMG_SIZE = 48

    def preprocess_image(self, img):
        """Resize, enhance contrast, and normalize the image for the model."""
        if len(img.shape) == 2:  # Image is already grayscale
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Image is in color
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unexpected image shape, cannot preprocess image.")
        
        # Apply histogram equalization to enhance contrast
        img = cv2.equalizeHist(img)
        
        # Normalize
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict_symbol(self, img):
        """Predict if the image contains a tick or a half-tick."""
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_class, confidence
    

    def convert_pdf_to_images(self, pdf_path):
        document = fitz.open(pdf_path)
        images = []
        headings = []
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append((img_data, page_num))
            headings.append(self.extract_headings(page))
        return images, headings
    
    def extract_headings(self, page):
        """Extract headings (e.g., Heading 1) from the page text using font size or style."""
        headings = []
        blocks = page.get_text("dict")["blocks"]  # Get text blocks in dictionary format

        for block in blocks:
            if "lines" in block:  # Ensure the block contains text lines
                for line in block["lines"]:
                    spans = line["spans"]
                    for span in spans:
                        text = span["text"].strip()
                        if len(text) > 0 and ("Heading 1" in text or span["size"] > 20):
                            headings.append(text)
                            print(f"Detected heading: {text}")
        
        return headings

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

    def segment_and_predict(self, img, show_plots):
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
                if confidence > 0.6:  # Adjust confidence threshold
                    predictions.append((predicted_class, confidence, (x, y, w, h)))
                    if show_plots:
                        plt.figure(figsize=(2, 2))
                        plt.imshow(symbol, cmap='gray')
                        plt.title(f'Predicted: {predicted_class}, Conf: {confidence:.2f}')
                        plt.axis('off')
                        plt.show()
        return predictions

    def predict_from_pdf(self, pdf_path, show_plots=False):
        ticks_per_heading = {}
        total_ticks = 0

        images, headings = self.convert_pdf_to_images(pdf_path)

        for img_data, page_num in images:
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("Could not decode image from PDF.")
                continue
            
            current_headings = headings[page_num]  # Get headings from the current page
            
            predictions = self.segment_and_predict(img, show_plots)
            for predicted_class, confidence, (x, y, w, h) in predictions:
                if predicted_class == 0:
                    total_ticks += 1
                    heading_key = current_headings[-1] if current_headings else f"Page {page_num + 1}"
                    if heading_key not in ticks_per_heading:
                        ticks_per_heading[heading_key] = 0
                    ticks_per_heading[heading_key] += 1
                    print(f"Tick detected with confidence {confidence:.2f} at position ({x}, {y}, {w}, {h})")
                elif predicted_class == 1:
                    total_ticks += 0.5
                    print(f"Half-tick detected with confidence {confidence:.2f} at position ({x}, {y}, {w}, {h})")
                    heading_key = current_headings[-1] if current_headings else f"Page {page_num + 1}"
                    if heading_key not in ticks_per_heading:
                        ticks_per_heading[heading_key] = 0
                    ticks_per_heading[heading_key] += 1

        print(f"Total ticks detected: {total_ticks}")
        for heading, count in ticks_per_heading.items():
            print(f"{heading}: {count} tick(s)")

        return total_ticks

