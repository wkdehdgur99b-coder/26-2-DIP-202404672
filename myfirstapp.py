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
    
    # 1. 이미지 로드 및 그레이스케일 변환 + 가우시안 블러(질감/노이즈 억제)
    img = Image.open(my_upload)
    f_raw = np.array(img.convert('L'), dtype=np.float32)
    f = nd.gaussian_filter(f_raw, sigma=2.0)
    h, w = f.shape
    total = h * w

    # 2. 배경 밝기 자동 판별 (테두리 픽셀 평균이 밝으면 흰 배경+어두운 객체로 판단하여 반전)
    border_pixels = np.concatenate([f[0, :], f[-1, :], f[:, 0], f[:, -1]])
    if np.mean(border_pixels) > 127:
        f = 255.0 - f

    # 3. Otsu 이진화 (전체 밝기 기반)
    f_uint8 = np.clip(f, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(f_uint8.ravel(), bins=256, range=(0, 256))
    current_max, threshold = 0, 0
    sum_total, sum_b, weight_b = np.dot(np.arange(256), hist), 0, 0

    for i in range(256):
        weight_b += hist[i]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += i * hist[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = i

    binary_img = f_uint8 > threshold

    # 4. 객체 내부 구멍 메우기 & 강한 형태학적 정리 (객체가 쪼개지는 현상 방지)
    binary_filled = nd.binary_fill_holes(binary_img)
    se_size = max(5, int(min(h, w) * 0.015))
    se_disk = np.ones((se_size, se_size), dtype=bool)
    binary_clean = nd.binary_opening(binary_filled, structure=se_disk)
    binary_clean = nd.binary_closing(binary_clean, structure=se_disk)

    # 5. 라벨링 및 의미 있는 크기의 객체만 필터링 (전체 면적의 0.3% 미만 노이즈 제거)
    labeled_array, num_features = nd.label(binary_clean, structure=np.ones((3, 3), dtype=int))
    min_pixels = int(total * 0.003)
    sizes = nd.sum(binary_clean, labeled_array, range(1, num_features + 1))
    valid_labels = np.where(sizes >= min_pixels)[0] + 1

    filtered_label = np.zeros_like(labeled_array)
    for new_idx, old_idx in enumerate(valid_labels, start=1):
        filtered_label[labeled_array == old_idx] = new_idx

    object_count = len(valid_labels)

    # 6. 최종 결과만 출력
    st.subheader("객체 검출 최종 결과")
    st.metric(label="검출된 총 객체 수", value=f"{object_count} 개")

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.cm.nipy_spectral.copy()
    cmap.set_under('black')
    ax.imshow(filtered_label, cmap=cmap, vmin=0.5)
    ax.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close(fig)

