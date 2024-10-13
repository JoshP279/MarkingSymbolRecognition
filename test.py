from SymbolRecognitionModel import SymbolRecognitionModel
import os
model_path = os.path.join(os.getcwd(), "left_handed_ticks.h5")
# model_path = os.path.join(os.getcwd(), "right_handed_ticks.h5")

symbol_model = SymbolRecognitionModel(model_path)

pdf_path = "submission_24.pdf"
ticks_detected, ticks_per_question = symbol_model.predict_from_pdf(pdf_path, show_plots=True)
print("Ticks per question:")
print(ticks_per_question)