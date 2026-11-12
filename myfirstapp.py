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

# ==========================================
# 4. 고도화된 DIP 결과 피드백 및 추천 감도 (맨 아래 추가)
# ==========================================
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
    _lum_arr = np.array(_proc.convert("L"), dtype=np.float64)
    _s = np.array(_proc.convert("HSV"))[:, :, 1]

    _mask = _s > sat_thresh
    if morph_size > 1:
        _mask = nd.binary_opening(_mask, structure=np.ones((morph_size, morph_size)))

    _labels, _raw_nums = nd.label(_mask)
    _areas = np.array([np.sum(_labels == i) for i in range(1, _raw_nums + 1)]) if _raw_nums > 0 else np.array([])

    # 메인 화면과 동일하게 Min Object Size 필터링 동기화
    if min_area > 1 and _raw_nums > 0:
        _valid = np.where(_areas >= min_area)[0] + 1
        _final_mask = np.isin(_labels, _valid)
        _labels, _nums = nd.label(_final_mask)
        _areas = np.array([np.sum(_labels == i) for i in range(1, _nums + 1)]) if _nums > 0 else np.array([])
    else:
        _final_mask = _mask
        _nums = _raw_nums

    # 1) Otsu + 배경 노이즈 바닥값을 결합한 추천 채도 임계값
    _hist, _ = np.histogram(_s.ravel(), bins=256, range=(0, 256))
    _tot = _s.size
    _sum_t = float(np.dot(np.arange(256), _hist))
    _sum_b, _w_b, _max_v, _otsu = 0.0, 0, 0.0, 50
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
            _max_v, _otsu = _v, _t

    _bg_s = _s[~_final_mask] if np.any(~_final_mask) else _s
    _noise_floor = int(np.percentile(_bg_s, 95)) + 12 if _bg_s.size > 0 else 40
    _rec_sat = int(np.clip(max(_otsu, _noise_floor), 20, 220))

    # 2) 전경(객체)과 배경 분리 분석 (흰 배경 오진 방지)
    _fg_lum = float(np.mean(_lum_arr[_final_mask])) if np.any(_final_mask) else float(np.mean(_lum_arr))
    _fg_sat = float(np.mean(_s[_final_mask])) if np.any(_final_mask) else 0.0
    _bg_sat = float(np.mean(_s[~_final_mask])) if np.any(~_final_mask) else 0.0
    _sat_gap = _fg_sat - _bg_sat

    _rec_bright = brightness
    if _fg_lum < 65:
        _rec_bright = float(np.clip(round(brightness + 0.3, 1), 0.2, 2.0))
    elif _fg_lum > 225 and _fg_sat < 60:
        _rec_bright = float(np.clip(round(brightness - 0.3, 1), 0.2, 2.0))

    _rec_blur = round(min(5.0, max(0.5, noise_level / 15.0)) * 2) / 2 if noise_level > 0 else 0.0
    _rec_morph = 5 if noise_level >= 25 else (3 if noise_level > 0 else 1)
    _noise_dots = int(np.sum(_areas < 35)) if _areas.size > 0 else 0
    _rec_min_area = 51 if (_noise_dots >= 3 or noise_level > 0) else 1

    # 3) 객체 형태 품질 분석 (뭉침 현상 & 내부 구멍)
    _filled = nd.binary_fill_holes(_final_mask)
    _hole_ratio = float(np.sum(_filled) - np.sum(_final_mask)) / max(float(np.sum(_filled)), 1.0)
    _merged_cnt = int(np.sum(_areas > np.median(_areas) * 3.2)) if _areas.size >= 3 else 0

    st.divider()
    st.write("### 4. DIP 영상 분석 피드백 & 추천 감도")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전경-배경 채도 분리도", f"{_sat_gap:.1f}", "분리 뚜렷함" if _sat_gap >= 70 else "분리 약함")
    c2.metric("추천 채도 임계값", f"{_rec_sat}", f"{_rec_sat - sat_thresh:+d}")
    c3.metric("추천 블러 / 모폴로지", f"{_rec_blur:.1f} / {_rec_morph}", f"현재 {blur_sigma:.1f} / {morph_size}")
    c4.metric("추천 최소 객체 크기", f"{_rec_min_area}", f"{_rec_min_area - min_area:+d}")

    tips = []

    # 채도 분리 및 임계값 진단
    if _nums == 0 or np.mean(_final_mask) < 0.005:
        tips.append(f"**[검출 부족]** 채도 임계값(`{sat_thresh}`)이 너무 높아 객체가 사라졌습니다. **{_rec_sat}** 부근으로 낮춰보세요.")
    elif np.mean(_final_mask) > 0.60 or abs(sat_thresh - _rec_sat) >= 25:
        tips.append(f"**[임계값 보정 필요]** 현재 채도 기준(`{sat_thresh}`)과 최적 분리 기준(**{_rec_sat}**)의 차이가 큽니다. 임계값을 조정하면 경계가 더 깔끔해집니다.")
    else:
        tips.append(f"**[채도 분리 양호]** 전경과 배경의 채도 차이가 `{_sat_gap:.1f}`로 뚜렷하며 임계값(`{sat_thresh}`)도 적절합니다.")

    # 객체 자체 밝기 진단
    if _fg_lum < 65:
        tips.append(f"**[객체 어두움]** 검출 영역의 평균 밝기(`{_fg_lum:.0f}`)가 낮아 채도 표현이 줄어들었습니다. 밝기를 **{_rec_bright:.1f}**로 높여보세요.")
    elif _fg_lum > 225 and _fg_sat < 60:
        tips.append(f"**[색상 날아감]** 객체 영역이 과노출(`{_fg_lum:.0f}`)되어 채도가 빠졌습니다. 밝기를 **{_rec_bright:.1f}**로 낮춰보세요.")

    # 노이즈 및 미세 파편 진단
    if noise_level > 0 and blur_sigma == 0 and morph_size == 1:
        tips.append(f"**[노이즈 미보정]** 노이즈(`{noise_level}`)가 추가되었으나 필터가 꺼져 있습니다. 블러 **{_rec_blur:.1f}**, 모폴로지 커널 **{_rec_morph}**을 적용해 보세요.")
    elif _noise_dots >= 3:
        tips.append(f"**[미세 노이즈 잔존]** 35픽셀 미만의 작은 점 **{_noise_dots}개**가 개수에 포함되었습니다. `최소 객체 픽셀 수`를 **{_rec_min_area}** 이상으로 올리세요.")

    # 객체 뭉침 및 내부 구멍 진단
    if _merged_cnt > 0:
        tips.append(f"**[객체 뭉침 감지]** 평균 크기보다 3배 이상 큰 덩어리가 **{_merged_cnt}개** 있습니다. 붙어 있는 객체가 합쳐졌을 가능성이 크므로 채도 임계값을 높이거나 모폴로지 커널을 키워보세요.")
    if _hole_ratio > 0.08:
        tips.append(f"**[객체 내부 구멍]** 객체 안쪽이 비어 있는 비율이 `{_hole_ratio*100:.1f}%`입니다. 가우시안 블러(`0.5~1.0`)를 살짝 주면 내부가 메워집니다.")

    for tip in tips:
        st.markdown(f"- {tip}")

