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

import streamlit as st
import numpy as np
import scipy.ndimage as nd
from PIL import Image
import matplotlib.pyplot as plt

# 1. 웹페이지 기본 설정
st.set_page_config(
    page_title="DIP Object Counter",
    page_icon="🔍",
    layout="centered"
)

# 2. 영상 처리 및 객체 검출 핵심 함수
def detect_and_count_objects(pil_image, obj_brightness="bright", min_area_ratio=0.0002):
    f = np.array(pil_image.convert('L'))
    h, w = f.shape

    if obj_brightness == "dark":
        f_proc = 255 - f
    else:
        f_proc = f.copy()

    # 불균일 조명 보정 (Top-hat 변환)
    radius = max(10, int(min(h, w) * 0.05))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    se_disk = x**2 + y**2 <= radius**2

    background = nd.grey_opening(f_proc, footprint=se_disk)
    f_tophat = np.clip(f_proc.astype(np.float32) - background.astype(np.float32), 0, 255).astype(np.uint8)

    # Otsu 이진화
    hist, _ = np.histogram(f_tophat.ravel(), bins=256, range=(0, 256))
    total = f_tophat.size
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

    binary_img = f_tophat > threshold

    # 형태학적 노이즈 제거
    se_3x3 = np.ones((3, 3), dtype=bool)
    binary_clean = nd.binary_opening(binary_img, structure=se_3x3)
    binary_clean = nd.binary_closing(binary_clean, structure=se_3x3)

    # Connected Component Labeling (8-연결성)
    labeled_array, num_features = nd.label(binary_clean, structure=np.ones((3, 3), dtype=int))

    # 작은 노이즈 필터링
    min_pixels = max(10, int(total * min_area_ratio))
    sizes = nd.sum(binary_clean, labeled_array, range(1, num_features + 1))
    valid_labels = np.where(sizes >= min_pixels)[0] + 1

    filtered_label = np.zeros_like(labeled_array)
    for new_idx, old_idx in enumerate(valid_labels, start=1):
        filtered_label[labeled_array == old_idx] = new_idx

    return len(valid_labels), filtered_label


    st.divider()
    st.header("⚙️ 검출 옵션 설정")
    brightness_option = st.radio(
        "객체와 배경의 밝기 관계",
        options=["bright", "dark"],
        format_func=lambda x: "어두운 배경 + 밝은 객체" if x == "bright" else "밝은 배경 + 어두운 객체"
    )
    sensitivity = st.slider(
        "노이즈 제거 민감도",
        min_value=0.0001,
        max_value=0.0020,
        value=0.0002,
        step=0.0001,
        format="%.4f"
    )

# --- 4. 메인 화면 UI (결과만 출력) ---
st.title("🔍 객체 검출 결과")

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    with st.spinner("분석 중..."):
        object_count, result_label = detect_and_count_objects(
            input_image,
            obj_brightness=brightness_option,
            min_area_ratio=sensitivity
        )

    # 최종 개수 표시
    st.metric(label="검출된 총 객체 수", value=f"{object_count} 개")

    # 원본과 결과 이미지 나란히 출력
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("원본 이미지")
        st.image(input_image, use_container_width=True)

    with col2:
        st.subheader("검출 결과")
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.nipy_spectral.copy()
        cmap.set_under('black')
        ax.imshow(result_label, cmap=cmap, vmin=0.5)
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close(fig)
    st.info("👆 위 버튼을 눌러 `rice.tif` 또는 테스트할 이미지를 업로드해 주세요.")
