import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="MY DIP")
st.write("## Homework 5 장동혁")
st.write(":dog:")

# ==========================================
# 사이드바 설정 (이미지 업로드 & DIP 파라미터 조절)
# ==========================================
st.sidebar.write("## my sidebar")
my_upload = st.sidebar.file_uploader("Upload", type=["png", "jpg", "jpeg"])

st.sidebar.divider()
st.sidebar.write("### 1. 밝기 및 대비 (Point Processing)")
brightness = st.sidebar.slider("밝기 (Brightness)", 0.2, 2.0, 1.0, step=0.1)
contrast = st.sidebar.slider("대비 (Contrast)", 0.2, 2.0, 1.0, step=0.1)

st.sidebar.write("### 2. 노이즈 및 필터링 (Spatial Filtering)")
noise_level = st.sidebar.slider("가우시안 노이즈 추가 (Noise)", 0, 50, 0, step=5)
blur_sigma = st.sidebar.slider("가우시안 스무딩 (Blur Sigma)", 0.0, 5.0, 0.0, step=0.5)

st.sidebar.write("### 3. 이진화 및 객체 검출 (Segmentation)")
sat_thresh = st.sidebar.slider("채도 임계값 (Saturation Threshold)", 0, 255, 50, step=1)
morph_size = st.sidebar.slider("노이즈 제거 커널 (Morphological Opening)", 1, 15, 1, step=2)
min_area = st.sidebar.slider("최소 객체 픽셀 수 (Min Object Size)", 1, 500, 1, step=10)


def process_image(upload):
    # 원본 불러오기 (RGB 통일)
    image = Image.open(upload).convert("RGB")

    # 1) 밝기 / 대비 조절
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    # 2) 가우시안 노이즈 추가 & 스무딩 필터 적용
    img_arr = np.array(image, dtype=np.float64)
    if noise_level > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_level, img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255)
    if blur_sigma > 0:
        img_arr = nd.gaussian_filter(img_arr, sigma=(blur_sigma, blur_sigma, 0))

    processed_img = Image.fromarray(img_arr.astype(np.uint8))

    # 상단 4개 컬럼 출력
    col1.write("Input (Adjusted)")
    col1.image(processed_img)

    gray_img = processed_img.convert('L')
    col2.write("Output 1: grayscale")
    col2.image(gray_img)

    histeq_img = ImageOps.equalize(gray_img)
    col3.write("Output 2: histeq")
    col3.image(histeq_img)

    col4.write("Output 3: saturation")
    hsv_image = processed_img.convert("HSV")
    s_channel = np.array(hsv_image)[:, :, 1]

    # 이진화 및 모폴로지 연산(Opening)
    binary_mask = s_channel > sat_thresh
    if morph_size > 1:
        binary_mask = nd.binary_opening(binary_mask, structure=np.ones((morph_size, morph_size)))

    output = (binary_mask * 255).astype(np.uint8)
    col4.image(output)

    # 3) 객체 라벨링 및 크기(픽셀 수) 계산
    labels, nums = nd.label(binary_mask)

    cc = np.zeros((nums, 2), dtype=int)
    for i in range(1, nums + 1):
        cc[i - 1, 0] = i
        cc[i - 1, 1] = np.sum(labels == i)

    # 최소 픽셀 수 미만인 자잘한 노이즈 객체 제거
    if min_area > 1 and nums > 0:
        valid_labels = cc[cc[:, 1] >= min_area, 0]
        filtered_mask = np.isin(labels, valid_labels)
        labels, nums = nd.label(filtered_mask)

        cc = np.zeros((nums, 2), dtype=int)
        for i in range(1, nums + 1):
            cc[i - 1, 0] = i
            cc[i - 1, 1] = np.sum(labels == i)

    # 하단 라벨링 결과 출력 (화면 비율을 1:2로 나누어 작고 깔끔하게 표시)
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.write(f"### Object Count: {nums}")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(labels, cmap='Spectral_r')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


col1, col2, col3, col4 = st.columns(4)

if my_upload is not None:
    process_image(upload=my_upload)
else:
    st.info("왼쪽 사이드 바에서 이미지 파일을 입력해주세요.")


if my_upload is not None:
    my_upload.seek(0)
    _img = Image.open(my_upload).convert("RGB")
    if brightness != 1.0:
        _img = ImageEnhance.Brightness(_img).enhance(brightness)
    if contrast != 1.0:
        _img = ImageEnhance.Contrast(_img).enhance(contrast)

    _arr = np.array(_img, dtype=np.float64)
    if noise_level > 0:
        _arr = np.clip(_arr + np.random.default_rng(42).normal(0, noise_level, _arr.shape), 0, 255)
    if blur_sigma > 0:
        _arr = nd.gaussian_filter(_arr, sigma=(blur_sigma, blur_sigma, 0))

    _proc = Image.fromarray(_arr.astype(np.uint8))
    _lum = float(np.mean(np.array(_proc.convert("L"))))
    _s = np.array(_proc.convert("HSV"))[:, :, 1]
    _mask = _s > sat_thresh
    if morph_size > 1:
        _mask = nd.binary_opening(_mask, structure=np.ones((morph_size, morph_size)))
    _, _nums = nd.label(_mask)

    # 추천 감도 계산 (Otsu 자동 채도 기준, 적정 밝기, 노이즈 보정치)
    _hist, _ = np.histogram(_s.ravel(), bins=256, range=(0, 256))
    _tot = _s.size
    _sum_t = float(np.dot(np.arange(256), _hist))
    _sum_b, _w_b, _max_v, _rec_sat = 0.0, 0, 0.0, 50
    for _t in range(256):
        _w_b += int(_hist[_t])
        if _w_b == 0:
            continue
        _w_f = _tot - _w_b
        if _w_f == 0:
            break
        _sum_b += _t * int(_hist[_t])
        _v = _w_b * _w_f * ((_sum_b / _w_b) - ((_sum_t - _sum_b) / _w_f)) ** 2
        if _v > _max_v:
            _max_v, _rec_sat = _v, _t

    _rec_bright = brightness if 105 <= _lum <= 155 else float(np.clip(round(brightness * (130.0 / max(_lum, 15.0)), 1), 0.2, 2.0))
    _rec_blur = round(min(5.0, max(0.5, noise_level / 15.0)) * 2) / 2 if noise_level > 0 else 0.0
    _rec_morph = 5 if noise_level >= 25 else (3 if noise_level > 0 else 1)

    st.divider()
    st.write("### 결과 피드백 & 추천 감도")
    st.info(
        f"**추천 감도** — 밝기: `{_rec_bright:.1f}` | 블러: `{_rec_blur:.1f}` | "
        f"채도 임계값: `{_rec_sat}` | 모폴로지 커널: `{_rec_morph}`"
    )

    tips = []
    if _lum < 85:
        tips.append(f"영상이 어둡습니다. 밝기를 **{_rec_bright:.1f}** 정도로 높여보세요.")
    elif _lum > 185:
        tips.append(f"영상이 너무 밝아 색이 날아갔습니다. 밝기를 **{_rec_bright:.1f}** 정도로 낮춰보세요.")

    if _nums == 0 or np.mean(_mask) < 0.005:
        tips.append(f"채도 임계값({sat_thresh})이 너무 높아 객체가 안 잡힙니다. **{_rec_sat}** 부근으로 낮춰보세요.")
    elif np.mean(_mask) > 0.65:
        tips.append(f"채도 임계값({sat_thresh})이 낮아 배경까지 잡힙니다. **{_rec_sat}** 부근으로 높여보세요.")

    if noise_level > 0 and blur_sigma == 0 and morph_size == 1:
        tips.append(f"노이즈가 있습니다. 블러(**{_rec_blur:.1f}**)나 모폴로지 커널(**{_rec_morph}**)을 적용해 보세요.")

    if not tips:
        tips.append("현재 밝기, 채도, 노이즈 설정이 균형 있게 잘 잡혀 있습니다.")

    for tip in tips:
        st.write(f"- {tip}")
