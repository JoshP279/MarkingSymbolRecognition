import os
import sys
from MNISTModel import MNISTModel  # If you need this in the future
from SymbolRecognitionModel import SymbolRecognitionModel

def main(pdf_path, submission_id, show_plots=False):
    model_path = "C:\\Users\\Joshua\\MarkingSymbolRecognition\\tick_detection_model.h5"  # Use absolute path
    symbol_model = SymbolRecognitionModel(model_path)
    
    # Mark the submission
    symbol_model.predict_from_pdf(pdf_path, show_plots=show_plots)

    # You can also include code to update the database or log the results as needed
    print(f"Finished processing submission ID: {submission_id}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script.py <pdf_path> <submission_id> [--show-plots]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    submission_id = sys.argv[2]
    show_plots = "--show-plots" in sys.argv
    # pdf_path = "submission_9.pdf"
    # submission_id = pdf_path.split("_")[1].split(".")[0]
    # show_plots = True
    main(pdf_path, submission_id, show_plots=show_plots)
