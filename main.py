from MNISTModel import MNISTModel
from SymbolRecognitionModel import SymbolRecognitionModel
import os

# mnist_model = MNISTModel()

# model_path = "mnist_model.h5"
# if os.path.exists(model_path):
#     mnist_model.load_model(model_path)
# else:
#     mnist_model.build_model()
#     mnist_model.train_model()
#     mnist_model.save_model(model_path)

# # Predict from PDF and optionally save or display images
# mnist_model.predict_from_pdf("submission_5.pdf")

# Path to the trained Keras model
model_path = "symbol_recognition_model.h5"

# Initialize the symbol recognition model
symbol_model = SymbolRecognitionModel(model_path)

# Predict from a PDF file and optionally save or display images
pdf_file = "submission_7.pdf"
symbol_model.predict_from_pdf(pdf_file)
