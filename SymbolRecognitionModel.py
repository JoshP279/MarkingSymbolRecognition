import fitz  
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

class SymbolRecognitionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.IMG_SIZE = 48
        self.ticks_per_heading = {}
        self.total_ticks = 0
        self.threshold = 0

    def preprocess_image(self, img):
        if len(img.shape) == 2:
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unexpected image shape.")
        img = cv2.equalizeHist(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    def predict_symbol(self, img):
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_class, confidence

    def extract_bookmarks(self, doc):
        """Extract bookmarks (outlines) from the PDF document."""
        bookmarks = []
        toc = doc.get_toc()
        for entry in toc:
            level, title, page_num = entry
            bookmarks.append({
                'title': title,
                'page_num': page_num
            })
        return bookmarks

    def segment_and_predict(self, img, show_plots=False):
        """Detect and predict symbols in the image."""
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
                symbol = img[y:y + h, x:x + w]
                symbol = self.resize_and_pad(symbol, self.IMG_SIZE)
                predicted_class, confidence = self.predict_symbol(symbol)
                if predicted_class == 0:
                    pred = "Tick"
                elif predicted_class == 1:
                    pred = "Messy Tick"
                elif predicted_class == 2:
                    pred = "Half Tick"
                elif predicted_class == 3:
                    pred = "Messy Half Tick"
                else:
                    pred = "Non-Tick"
                predictions.append((predicted_class, confidence, (x, y, w, h)))
                if show_plots and confidence >= self.threshold:
                    plt.imshow(symbol, cmap='gray')
                    plt.title(f'Predicted: {pred}, Conf: {confidence:.2f}')
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

    def convert_pdf_to_images(self, pdf_path):
        """Convert PDF pages to images."""
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append((img_data, page_num))
        return images, doc

    def extract_headings_with_coordinates(self, page):
        """Extract text headings with coordinates on a PDF page."""
        headings = []
        blocks = page.get_text("dict")["blocks"]  
        for block in blocks:
            if "lines" in block:  
                for line in block["lines"]:
                    spans = line["spans"]
                    for span in spans:
                        text = span["text"].strip()
                        bbox = span["bbox"]  
                        if len(text) > 0 and span["size"] > 12:  
                            headings.append({
                                'text': text,
                                'bbox': bbox
                            })
        return headings

    def predict_from_pdf(self, pdf_path, show_plots=False):
        """Detect ticks from PDF and associate them with headings using coordinates."""
        doc = fitz.open(pdf_path)  
        images, doc = self.convert_pdf_to_images(pdf_path)
        
        total_ticks = 0
        for img_data, page_num in images:
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Could not decode image for page {page_num + 1}.")
                continue

            page = doc.load_page(page_num)
            headings = self.extract_headings_with_coordinates(page)  

            predictions = self.segment_and_predict(img, show_plots)

            for predicted_class, confidence, (x, y, w, h) in predictions:
                if confidence >= self.threshold:
                    nearest_heading = None
                    min_distance = float('inf')

                    for heading in headings:
                        heading_bbox = heading['bbox']
                        heading_y = heading_bbox[1] 
                        distance = abs(heading_y - y)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_heading = heading['text']

                    if nearest_heading:
                        if nearest_heading not in self.ticks_per_heading:
                            self.ticks_per_heading[nearest_heading] = 0
                        if predicted_class in [0, 1]:
                            self.ticks_per_heading[nearest_heading] += 1
                            total_ticks += 1
                        elif predicted_class in [2, 3]:
                            self.ticks_per_heading[nearest_heading] += 0.5
                            total_ticks += 0.5
                    else:
                        total_ticks += 1 if predicted_class in [0, 1] else 0.5

        print(f"Total ticks detected: {total_ticks}")
        for heading, count in sorted(self.ticks_per_heading.items()):
            print(f"{heading}: {count} tick(s)")
        return total_ticks, self.ticks_per_heading