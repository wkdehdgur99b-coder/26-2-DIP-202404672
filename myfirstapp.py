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
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy import ndimage as nd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Homework 2 장동혁")
st.write(":dog:")
st.sidebar.write("## my sidebar")

# 사이드바에서 임계값(Threshold)과 커널 크기를 조절할 수 있도록 추가 (선택사항)
thresh_val = st.sidebar.slider("Threshold (임계값)", 0, 255, 64)
kernel_size = st.sidebar.slider("Opening Kernel Size (노이즈 제거 크기)", 1, 25, 9, step=2)

def process_image(upload):
    # 1. 원본 및 그레이스케일 변환
    image = Image.open(upload)
    gray_img = image.convert('L')
    f = np.array(gray_img)
    
    col1.write("1. Input (Grayscale)")
    col1.image(gray_img, use_container_width=True)
    
    # 2. 이진화 (Thresholding: 어두운 물체 찾기)
    A = f < thresh_val
    col2.write(f"2. Binary (f < {thresh_val})")
    # Streamlit 출력을 위해 True/False를 0과 255로 변환
    col2.image((A * 255).astype(np.uint8), use_container_width=True)
    
    # 3. 모폴로지 연산 (Binary Opening으로 작은 노이즈 제거)
    A_e = nd.binary_opening(A, np.ones((kernel_size, kernel_size)))
    col3.write(f"3. Opening ({kernel_size}x{kernel_size})")
    col3.image((A_e * 255).astype(np.uint8), use_container_width=True)
    
    # 4. 객체 라벨링 (Connected Component Labeling)
    labeled_array, num_objects = nd.label(A_e)
    col4.write("4. Labeled Objects")
    
    # 라벨링 결과를 tab20c 컬러맵으로 시각화
    cmap = plt.get_cmap('tab20c')
    # 배경(0)은 검은색으로, 나머지 객체는 컬러로 표시
    colored_labels = cmap(labeled_array % 20)[:, :, :3]
    colored_labels[labeled_array == 0] = 0
    col4.image(colored_labels, use_container_width=True)
    
    # 5. 최종 결과 (Object 개수) 화면에 크게 출력
    st.success(f"🔍 이미지 안에서 찾은 Object의 개수는 총 **{num_objects}개**입니다!")

col1, col2, col3, col4 = st.columns(4)

# 사이드바 파일 업로더
my_upload = st.sidebar.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    process_image(my_upload)
else:
    st.info("👈 왼쪽 사이드바에서 분석할 이미지를 업로드해 주세요.")
