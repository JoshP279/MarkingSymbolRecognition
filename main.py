import os
import sys
import requests
from SymbolRecognitionModel import SymbolRecognitionModel

def update_submission_mark(server_url, submission_id, total_mark):
    server_url += "/updateSubmissionMark"
    data = {
        "submissionID": submission_id,
        "totalMark": total_mark
    }

    try:
        response = requests.put(server_url, json=data)
        response.raise_for_status()
        print(f"Submission ID: {submission_id}, Total Mark: {total_mark}")
        print(f"Successfully updated submission ID {submission_id} on the server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update submission ID {submission_id} on the server. Error: {e}")

def update_question_mark(server_url, submission_id, question_id, mark_allocation):
    server_url += "/updateQuestionMark"
    data = {
        "submissionID": submission_id,
        "questionID": question_id,
        "markAllocation": mark_allocation
    }

    try:
        response = requests.put(server_url, json=data)
        response.raise_for_status()
        print(f"Submission ID: {submission_id}, Question ID: {question_id}, Mark Allocation: {mark_allocation}")
        print(f"Successfully updated question ID {question_id} with mark allocation on the server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update question ID {question_id} for submission ID {submission_id}. Error: {e}")

def main(pdf_path, submission_id, TotalMark, MarkingStyle, show_plots=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MarkingStyle = MarkingStyle.strip().strip('"')

    if MarkingStyle == "Right Handed Ticks":
        model_path = os.path.join(script_dir, "right_handed_ticks.h5")
    else:
        model_path = os.path.join(script_dir, "left_handed_ticks.h5")

    print(model_path)
    symbol_model = SymbolRecognitionModel(model_path)
    
    ticks_detected, ticks_per_question = symbol_model.predict_from_pdf(pdf_path, show_plots=show_plots)
    print("Ticks per question:")
    print(ticks_per_question)
    if ticks_detected is None:
        print(f"No ticks detected for submission ID {submission_id}.")
        total_mark = 0
    else:
        total_mark = round((float(ticks_detected) / float(TotalMark)) * 100,2)
        total_mark = max(0, min(total_mark, 100))

    server_url = "http://192.168.202.75:8080"
    update_submission_mark(server_url, submission_id, total_mark)
    for question_id, mark_allocation in ticks_per_question.items():
        update_question_mark(server_url, submission_id, question_id, mark_allocation)
        
    print(f"Finished processing submission ID: {submission_id}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: script.py <pdf_path> <submission_id> <total_marks> <marking_style> [--show-plots]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    submission_id = sys.argv[2]
    TotalMarks = sys.argv[3]
    MarkingStyle = sys.argv[4]
    show_plots = "--show-plots" in sys.argv

    main(pdf_path, submission_id, TotalMarks, MarkingStyle, show_plots=show_plots)
