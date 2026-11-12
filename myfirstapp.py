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
    
    # 1. HSV 채도(Saturation) 채널로 모든 색상의 레고 추출
    img = Image.open(my_upload).convert("RGB")
    hsv_image = img.convert("HSV")
    s_channel = np.array(hsv_image)[:, :, 1]
    
    binary_img = s_channel > 50
    binary_filled = nd.binary_fill_holes(binary_img)
    se_5x5 = np.ones((5, 5), dtype=bool)
    binary_clean = nd.binary_opening(binary_filled, structure=se_5x5)
    binary_clean = nd.binary_closing(binary_clean, structure=se_5x5)

    # 2. 거리 변환 후 가우시안 스무딩으로 레고 돌기(스터드) 노이즈 제거
    dist_transform = nd.distance_transform_edt(binary_clean)
    dist_smooth = nd.gaussian_filter(dist_transform, sigma=3.0)

    # 블록 반지름 크기에 맞춘 지역 최댓값(중심점) 탐색
    local_max_size = max(17, int(min(binary_clean.shape) * 0.05))
    local_maxi = (dist_smooth == nd.maximum_filter(dist_smooth, size=local_max_size)) & (dist_smooth > local_max_size * 0.35)
    
    # 아주 가까이 붙은 중복 중심점 하나로 병합
    local_maxi = nd.binary_dilation(local_maxi, structure=np.ones((5, 5), dtype=bool))
    markers, object_count = nd.label(local_maxi)

    # 3. 각 레고 영역을 가장 가까운 중심점 기준으로 깔끔하게 분할
    _, nearest_idx = nd.distance_transform_edt(markers == 0, return_indices=True)
    segmented_label = markers[tuple(nearest_idx)] * binary_clean

    # 4. 각 객체마다 뚜렷하게 구분되는 랜덤 색상 입히기
    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(object_count + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # 배경은 검은색
    colored_result = colors[segmented_label]

    # 5. 최종 결과 출력
    st.subheader("객체 검출 최종 결과")
    st.metric(label="검출된 총 객체 수", value=f"{object_count} 개")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(colored_result)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)

