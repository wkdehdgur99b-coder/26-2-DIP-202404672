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
    # 그레이스케일 변환 및 NumPy 배열화
    f = np.array(pil_image.convert('L'))
    h, w = f.shape

    # 어두운 객체(흰 배경)인 경우 영상 반전하여 통일된 로직으로 처리
    if obj_brightness == "dark":
        f_proc = 255 - f
    else:
        f_proc = f.copy()

    # 이미지 크기에 비례한 구조 요소(Structuring Element) 크기 설정 (불균일 조명 보정)
    radius = max(10, int(min(h, w) * 0.05))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    se_disk = x**2 + y**2 <= radius**2

    # Top-hat 변환 (배경 추정 및 제거)
    background = nd.grey_opening(f_proc, footprint=se_disk)
    f_tophat = np.clip(f_proc.astype(np.float32) - background.astype(np.float32), 0, 255).astype(np.uint8)

    # Otsu 이진화 알고리즘
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

    # 형태학적 연산(Opening & Closing)으로 미세 노이즈 제거 및 객체 내부 채우기
    se_3x3 = np.ones((3, 3), dtype=bool)
    binary_clean = nd.binary_opening(binary_img, structure=se_3x3)
    binary_clean = nd.binary_closing(binary_clean, structure=se_3x3)

    # Connected Component Labeling (8-연결성)
    labeled_array, num_features = nd.label(binary_clean, structure=np.ones((3, 3), dtype=int))

    # 전체 면적 대비 너무 작은 노이즈 필터링
    min_pixels = max(10, int(total * min_area_ratio))
    sizes = nd.sum(binary_clean, labeled_array, range(1, num_features + 1))
    valid_labels = np.where(sizes >= min_pixels)[0] + 1

    # 유효한 객체만 1번부터 순차적으로 재라벨링
    filtered_label = np.zeros_like(labeled_array)
    for new_idx, old_idx in enumerate(valid_labels, start=1):
        filtered_label[labeled_array == old_idx] = new_idx

    return len(valid_labels), filtered_label

# --- 3. 웹사이트 UI 구성 ---
st.title("🔍 이미지 객체 검출 및 카운팅")
st.write("이미지를 업로드하면 디지털 영상 처리(Top-hat 변환, Otsu 이진화, 라벨링)를 통해 객체의 개수를 자동으로 계산합니다.")

# 사이드바 옵션 (다양한 테스트 이미지 대응용)
with st.sidebar:
    st.header("⚙️ 검출 옵션 설정")
    brightness_option = st.radio(
        "객체와 배경의 밝기 관계",
        options=["bright", "dark"],
        format_func=lambda x: "어두운 배경 + 밝은 객체 (예: rice.tif)" if x == "bright" else "밝은 배경 + 어두운 객체 (예: 동전, 점)"
    )
    sensitivity = st.slider(
        "노이즈 제거 민감도 (작은 점 무시 정도)",
        min_value=0.0001,
        max_value=0.0020,
        value=0.0002,
        step=0.0001,
        format="%.4f"
    )

# 이미지 파일 업로더
uploaded_file = st.file_uploader(
    "분석할 이미지를 선택하세요 (TIF, PNG, JPG, JPEG)",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # 업로드된 이미지 열기
    input_image = Image.open(uploaded_file)

    # 객체 검출 수행
    with st.spinner("이미지에서 객체를 분석하고 있습니다..."):
        object_count, result_label = detect_and_count_objects(
            input_image,
            obj_brightness=brightness_option,
            min_area_ratio=sensitivity
        )

    st.divider()

    # 최종 개수 강조 표시
    st.metric(label="검출된 총 객체 수", value=f"{object_count} 개")

    # 원본 이미지와 최종 검출 결과 이미지를 나란히 배치
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("원본 이미지")
        st.image(input_image, use_container_width=True)

    with col2:
        st.subheader("객체 검출 결과")
        # 배경(0)은 검은색, 각 객체는 서로 다른 색상으로 표시
        fig, ax = plt.subplots(figsize=(5, 5))
        cmap = plt.cm.nipy_spectral.copy()
        cmap.set_under('black')
        ax.imshow(result_label, cmap=cmap, vmin=0.5)
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("👆 위 버튼을 눌러 `rice.tif` 또는 테스트할 이미지를 업로드해 주세요.")
