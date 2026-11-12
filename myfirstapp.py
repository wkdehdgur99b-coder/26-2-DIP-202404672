import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Homework 2 장동혁")
st.write(":dog:")
st.sidebar.write("## my sidebar")

def process_image(upload):
    image = Image.open(upload)
    col1.write("Input")
    col1.image(image)
    
    output = image.convert('L')
    col2.write("Output 1: grayscale")
    col2.image(output)
    
    output = ImageOps.equalize(output)
    col3.write("Output 2: histeq")
    col3.image(output)
    
    # Output 3: Canny Edge Detection
    col4.write("Output 3: Canny Edge Detection")
    img_array = np.array(output)
    edges = cv2.Canny(img_array, 100, 100)
    col4.image(edges)

col1, col2, col3, col4 = st.columns(4)
my_upload = st.sidebar.file_uploader("Upload", type=["png","jpg","jpeg"])

if my_upload is not None:
    process_image(upload=my_upload)
else:
    process_image("./gr.jpeg")
