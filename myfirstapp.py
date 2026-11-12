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

import scipy.ndimage as nd
import matplotlib.pyplot as plt

if my_upload is not None:
    st.divider()
    
    # 1. 교수님 코드의 핵심인 HSV 채도(Saturation) 채널로 전경(모든 색상의 레고) 추출
    img = Image.open(my_upload).convert("RGB")
    hsv_image = img.convert("HSV")
    s_channel = np.array(hsv_image)[:, :, 1]
    
    # 채도가 높은 객체 영역 분리 + 구멍 메우기 + 노이즈 제거
    binary_img = s_channel > 50
    binary_filled = nd.binary_fill_holes(binary_img)
    se_3x3 = np.ones((3, 3), dtype=bool)
    binary_clean = nd.binary_opening(binary_filled, structure=se_3x3)
    binary_clean = nd.binary_closing(binary_clean, structure=se_3x3)

    # 2. 거리 변환(Distance Transform)으로 붙어 있는 레고들의 중심점(Seed) 분리
    dist_transform = nd.distance_transform_edt(binary_clean)
    # 블록 크기에 비례한 탐색 반경 설정
    local_max_size = max(15, int(min(binary_clean.shape) * 0.045))
    local_maxi = (dist_transform == nd.maximum_filter(dist_transform, size=local_max_size)) & (dist_transform > local_max_size * 0.35)
    
    # 3. 분리된 중심점 마커 라벨링 후 Watershed 방식으로 각 영역 분할
    markers, object_count = nd.label(local_maxi)
    # 각 전경 픽셀을 가장 가까운 마커(레고 중심)에 할당하여 시각적으로도 깔끔하게 분리
    _, nearest_idx = nd.distance_transform_edt(markers == 0, return_indices=True)
    segmented_label = markers[tuple(nearest_idx)] * binary_clean

    # 4. 최종 결과만 출력
    st.subheader("객체 검출 최종 결과")
    st.metric(label="검출된 총 객체 수", value=f"{object_count} 개")

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.nipy_spectral.copy()
    cmap.set_under('black')
    ax.imshow(segmented_label, cmap=cmap, vmin=0.5)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)
   

