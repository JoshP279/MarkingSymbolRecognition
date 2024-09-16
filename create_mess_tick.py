import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import string

def generate_text_image(output_path, img_size=(48, 48), text_color=(0, 0, 0), font_size=8, num_lines=3):
    """
    Generate an image with random background text, with multiple lines.

    Args:
        output_path (str): Path to save the generated PNG image.
        img_size (tuple): Size of the generated image, default is (48, 48).
        text_color (tuple): Color of the text, default is black (0, 0, 0).
        font_size (int): Font size of the text.
        num_lines (int): Number of lines of random text to generate.
    """
    # Create a blank white image
    img = Image.new('RGB', img_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use a basic font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default PIL font if TTF font is unavailable
        font = ImageFont.load_default()

    line_height = font.getbbox('A')[3]  # Get the height of a line
    total_text_height = line_height * num_lines

    # Start drawing text from the top
    y_position = (img_size[1] - total_text_height) // 2

    for _ in range(num_lines):
        random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        # Calculate text width and center it horizontally
        bbox = draw.textbbox((0, 0), random_text, font=font)
        text_width = bbox[2] - bbox[0]
        x_position = (img_size[0] - text_width) // 2

        # Draw the text onto the image
        draw.text((x_position, y_position), random_text, fill=text_color, font=font)

        # Move to the next line
        y_position += line_height

    # Save the image as a PNG
    img.save(output_path)
    print(f"Generated image with background text saved at {output_path}")


# Example usage:
output_file = "text_background_multiline.png"
generate_text_image(output_file, img_size=(48, 48), text_color=(0, 0, 0), font_size=8, num_lines=3)
