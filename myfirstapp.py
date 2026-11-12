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

import streamlit as st
from PIL import Image

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 파일을 바로 열어서 흑백으로 변환
    f = Image.open(uploaded_file).convert('L')
    st.image(f, caption="흑백 변환 이미지")

f = Image.open('myimg.jpg').convert('L')
f = np.array(f)
plt.imshow(f,cmap='gray')
plt.axis('off')
plt.show()

A = f<64
plt.imshow(A,cmap='gray')
plt.axis('off')
plt.show()

A_e = nd.binary_opening(A,np.ones((9,9)))
plt.imshow(A_e,cmap='gray')
plt.axis('off')
plt.show()

l, m = nd.label(A_e)
plt.imshow(l,interpolation='nearest',cmap='tab20c')
plt.axis('off')
plt.show()
print(f'  {m}개가 맞나요?')
