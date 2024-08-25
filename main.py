import os
import sys
import requests  # Import the requests library to make HTTP requests
from SymbolRecognitionModel import SymbolRecognitionModel

def update_submission_mark(server_url, submission_id, total_mark):
    data = {
        "submissionID": submission_id,
        "totalMark": total_mark
    }

    try:
        response = requests.put(server_url, json=data)  # You can use POST or PUT depending on your server setup
        response.raise_for_status()  # Raise an HTTPError for bad responses
        print(f"Successfully updated submission ID {submission_id} on the server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update submission ID {submission_id} on the server. Error: {e}")

def main(pdf_path, submission_id, TotalMark, show_plots=False):
    model_path = "C:\\MarkingSymbolRecognition\\tick_detection_model.h5"
    symbol_model = SymbolRecognitionModel(model_path)
    
    # Mark the submission and get the number of ticks detected
    ticks_detected = symbol_model.predict_from_pdf(pdf_path, show_plots=show_plots)
    
    if ticks_detected is None:
        print(f"No ticks detected for submission ID {submission_id}.")
        total_mark = 0
    else:
        # Replace this with your actual logic for calculating total marks based on assessment_id
        # Assuming total_mark is calculated based on ticks_detected (example logic here)
        total_mark = float(ticks_detected) / float(TotalMark) * 100  # Ensure both are floats for division

        # Server URL to update the submission mark (replace with your actual server endpoint)
        server_url = "http://10.112.49.6:3306/updateSubmissionMark"  # Replace 5000 with the correct port number
        
        # Update the submission mark on the server
        update_submission_mark(server_url, submission_id, total_mark)

    print(f"Finished processing submission ID: {submission_id}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: script.py <pdf_path> <submission_id> <assessment_id> [--show-plots]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    submission_id = sys.argv[2]
    TotalMarks = sys.argv[3]
    show_plots = "--show-plots" in sys.argv
    
    main(pdf_path, submission_id, TotalMarks, show_plots=show_plots)
