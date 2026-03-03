import cv2
from PIL import Image

def save_uploaded_image(uploaded_file, save_path="temp.jpg"):
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image.save(save_path)
    return save_path

def bgr_to_rgb(image):
    # convert image brg to rgb color.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

