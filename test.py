from SymbolRecognitionModel import SymbolRecognitionModel
# model_path = "C:\\MarkingSymbolRecognition\\left_handed_ticks.h5" # honours lab pc

# model_path = "C:\\Users\\Joshua\\MarkingSymbolRecognition\\left_handed_ticks.h5" # josh's pc
model_path = "C:\\Users\\Joshua\\MarkingSymbolRecognition\\right_handed_ticks.h5" # josh's pc

symbol_model = SymbolRecognitionModel(model_path)
pdf_path = "submission_1.pdf"
ticks_detected, ticks_per_question = symbol_model.predict_from_pdf(pdf_path, show_plots=False)
print("Ticks per question:")
print(ticks_per_question)