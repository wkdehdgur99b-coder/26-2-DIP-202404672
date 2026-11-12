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

# ==========================================
# 2. 배경 불균일 보정 (Background Correction)
# ==========================================
background = nd.grey_opening(img, size=(15, 15))
img_corrected = img - background

# ==========================================
# 3. 이진화 (Thresholding)
# ==========================================
threshold = 50
binary = img_corrected > threshold

# ==========================================
# 4. 노이즈 제거 (Morphological Filtering)
# ==========================================
cleaned = nd.binary_opening(binary, structure=np.ones((3, 3)))

# ==========================================
# 5. 객체 라벨링 및 개수 세기 (Object Counting)
# ==========================================
labeled_array, num_objects = nd.label(cleaned)
print(f"검출된 총 객체(쌀알) 개수: {num_objects}개")

# ==========================================
# 6. 전체 과정 시각화 (Visualization)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('1. Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(background, cmap='gray')
axes[0, 1].set_title('2. Estimated Background')
axes[0, 1].axis('off')

axes[0, 2].imshow(img_corrected, cmap='gray')
axes[0, 2].set_title('3. Background Subtracted')
axes[0, 2].axis('off')

axes[1, 0].imshow(binary, cmap='gray')
axes[1, 0].set_title(f'4. Thresholding (> {threshold})')
axes[1, 0].axis('off')

axes[1, 1].imshow(cleaned, cmap='gray')
axes[1, 1].set_title('5. Morphological Opening')
axes[1, 1].axis('off')

axes[1, 2].imshow(labeled_array, cmap='nipy_spectral')
axes[1, 2].set_title(f'6. Labeled Objects (Count: {num_objects})')
axes[1, 2].axis('off')

plt.show()
