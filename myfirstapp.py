import streamlit as st
from PIL import Image, ImageOps
import numpy as np


st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Homework 5 장동혁")
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
    col4.write("Output 3:saturation")
    hsv_image = image.convert("HSV")
    s_channel = np.array(hsv_image)[:, :, 1]
    output = np.where(s_channel > 50, 255, 0).astype(np.uint8)
    col4.image(output)

col1, col2, col3, col4 = st.columns(4)
my_upload = st.sidebar.file_uploader("Upload", type=["png","jpg","jpeg"])

if my_upload is not None:
    process_image(upload=my_upload)
else:
    st.info("왼쪽 사이드 바에서 이미지 파일을 입력해주세요.")

import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.ndimage as nd
from PIL import Image

st.set_page_config(page_title="DIP Object Counting", layout="wide")
st.title("DIP Object Counting")

# 1. 이미지 불러오기 (Load Image)
uploaded_file = st.file_uploader(
    "이미지 파일 업로드",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

default_path = None
for root, _, files in os.walk("."):
    if "rice.tif" in files:
        default_path = os.path.join(root, "rice.tif")
        break

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
elif default_path is not None:
    image = Image.open(default_path).convert("L")
else:
    st.info("위 버튼을 눌러 분석할 이미지를 업로드해 주세요.")
    st.stop()

img = np.array(image, dtype=np.float64)

# 파라미터 설정
use_bg_sub = st.checkbox("배경 불균일 보정 적용 (Grey Opening)", value=True)

col1, col2, col3 = st.columns(3)
with col1:
    bg_size = st.slider("배경 제거 커널 크기", 3, 51, 15, step=2)
with col2:
    threshold = st.slider("이진화 임계값 (Threshold)", 0, 255, 50, step=1)
with col3:
    morph_size = st.slider("노이즈 제거 커널 크기", 1, 15, 3, step=2)

# 2. 배경 불균일 보정 (Background Correction)
if use_bg_sub:
    background = nd.grey_opening(img, size=(bg_size, bg_size))
    img_processed = img - background
else:
    background = np.zeros_like(img)
    img_processed = img

# 3. 이진화 (Thresholding)
binary = img_processed > threshold

# 4. 노이즈 제거 (Morphological Filtering)
cleaned = nd.binary_opening(binary, structure=np.ones((morph_size, morph_size)))

# 5. 객체 라벨링 및 개수 세기 (Object Counting)
labeled_array, num_objects = nd.label(cleaned)
st.subheader(f"검출된 객체 수: {num_objects}")

# 6. 결과 시각화 (Visualization)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("1. Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(background, cmap="gray")
axes[0, 1].set_title("2. Estimated Background")
axes[0, 1].axis("off")

axes[0, 2].imshow(img_processed, cmap="gray")
axes[0, 2].set_title("3. Processed Image")
axes[0, 2].axis("off")

axes[1, 0].imshow(binary, cmap="gray")
axes[1, 0].set_title(f"4. Thresholding (> {threshold})")
axes[1, 0].axis("off")

axes[1, 1].imshow(cleaned, cmap="gray")
axes[1, 1].set_title("5. Morphological Opening")
axes[1, 1].axis("off")

axes[1, 2].imshow(labeled_array, cmap="nipy_spectral")
axes[1, 2].set_title(f"6. Labeled Objects (Count: {num_objects})")
axes[1, 2].axis("off")

st.pyplot(fig)
plt.close(fig)

