import os

# Define the path to your folder
folder = "C:\\Users\\Joshua\\MarkingSymbolRecognition\\raw_data\\messy_half_ticks"

files = os.listdir(folder)

counter = 1

for filename in files:
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension == '.png':
        new_filename = f"image{counter}{file_extension}"
        
        while os.path.exists(os.path.join(folder, new_filename)):
            counter += 1
            new_filename = f"image{counter}{file_extension}"
        
        old_file = os.path.join(folder, filename)
        new_file = os.path.join(folder, new_filename)
        
        os.rename(old_file, new_file)
        
        counter += 1

print(f"Renamed {counter - 1} images in the folder.")
