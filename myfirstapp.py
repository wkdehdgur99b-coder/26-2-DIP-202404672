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

    # 하단 라벨링 결과 출력
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

    return s_channel, gray_img, binary_mask, nums, cc


def compute_otsu_threshold(channel):
    hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
    total = channel.size
    current_max, best_thresh = 0.0, 50
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b, w_b = 0.0, 0
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * int(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > current_max:
            current_max = var_between
            best_thresh = t
    return best_thresh


col1, col2, col3, col4 = st.columns(4)

if my_upload is not None:
    s_channel, gray_img, binary_mask, nums, cc = process_image(upload=my_upload)

    # ==========================================
    # 4. 결과 피드백 및 추천 감도 (추가된 하단 섹션)
    # ==========================================
    st.divider()
    st.write("### 4. 파라미터 결과 피드백 및 추천 감도")

    # (1) 추천 감도 계산
    mean_lum = float(np.mean(np.array(gray_img, dtype=np.float64)))
    rec_brightness = float(np.clip(round(brightness * (130.0 / max(mean_lum, 15.0)), 1), 0.2, 2.0))
    if 105 <= mean_lum <= 155:
        rec_brightness = brightness

    rec_sat = compute_otsu_threshold(s_channel)
    rec_blur = round(min(5.0, max(0.5, noise_level / 15.0)) * 2) / 2 if noise_level > 0 else 0.0
    rec_morph = 5 if noise_level >= 25 else (3 if noise_level > 0 else 1)

    tiny_objects = int(np.sum(cc[:, 1] < 30)) if nums > 0 else 0
    if tiny_objects >= 5 and rec_morph == 1:
        rec_morph = 3
    rec_min_area = 41 if (tiny_objects >= 5 or noise_level > 0) else 1

    # 추천 감도 수치 카드 출력
    st.write("#### 추천 감도 설정값")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("추천 밝기 (Brightness)", f"{rec_brightness:.1f}", f"{rec_brightness - brightness:+.1f}")
    m2.metric("추천 블러 (Blur Sigma)", f"{rec_blur:.1f}", f"{rec_blur - blur_sigma:+.1f}")
    m3.metric("추천 채도 기준 (Otsu)", f"{rec_sat}", f"{rec_sat - sat_thresh:+d}")
    m4.metric("추천 모폴로지 커널", f"{rec_morph}", f"{rec_morph - morph_size:+d}")
    m5.metric("추천 최소 객체 크기", f"{rec_min_area}", f"{rec_min_area - min_area:+d}")

    # (2) 항목별 상세 피드백 생성
    st.write("#### 현재 조절값 진단 피드백")
    feedbacks = []
    fg_ratio = float(np.mean(binary_mask) * 100.0)

    # 밝기 진단
    if mean_lum < 85:
        feedbacks.append(
            f"**[밝기 부족]** 현재 평균 휘도가 낮아(`{mean_lum:.1f}/255`) 어두운 영역의 채도 정보가 손실될 수 있습니다. "
            f"밝기를 **{rec_brightness:.1f}** 부근으로 높이는 것을 권장합니다."
        )
    elif mean_lum > 185:
        feedbacks.append(
            f"**[과다 노출]** 영상이 너무 밝아(`{mean_lum:.1f}/255`) 색상이 하얗게 날아가면서 채도(Saturation)가 급감했습니다. "
            f"밝기를 **{rec_brightness:.1f}** 부근으로 낮춰주세요."
        )
    else:
        feedbacks.append(
            f"**[밝기 적정]** 평균 휘도(`{mean_lum:.1f}/255`)가 안정적이어서 채도 분리 왜곡이 적습니다."
        )

    # 채도 임계값 진단
    if nums == 0 or fg_ratio < 0.3:
        feedbacks.append(
            f"**[채도 임계값 과다]** 임계값(`{sat_thresh}`)이 너무 높아 객체가 거의 검출되지 않았습니다(전경 비율 `{fg_ratio:.1f}%`). "
            f"Otsu 알고리즘 기반 추천값인 **{rec_sat}** 전후로 낮춰보세요."
        )
    elif fg_ratio > 65.0 or sat_thresh < rec_sat - 25:
        feedbacks.append(
            f"**[채도 임계값 부족]** 임계값(`{sat_thresh}`)이 낮아 배경 노이즈까지 전경에 포함되었습니다(전경 비율 `{fg_ratio:.1f}%`). "
            f"추천 기준값인 **{rec_sat}** 전후로 높이면 배경이 깔끔하게 분리됩니다."
        )
    else:
        feedbacks.append(
            f"**[채도 임계값 양호]** 현재 임계값(`{sat_thresh}`)이 영상의 최적 분리 기준(`{rec_sat}`)과 잘 맞습니다."
        )

    # 노이즈 및 스무딩 진단
    if noise_level > 0 and blur_sigma == 0.0 and morph_size == 1:
        feedbacks.append(
            f"**[노이즈 미보정]** 가우시안 노이즈(`{noise_level}`)가 추가된 상태에서 필터가 꺼져 있어 자잘한 점들이 객체로 오검출될 수 있습니다. "
            f"`Blur Sigma`를 **{rec_blur:.1f}**, `모폴로지 커널`을 **{rec_morph}**으로 설정해 보세요."
        )
    elif blur_sigma >= 3.0:
        feedbacks.append(
            f"**[과도한 스무딩]** `Blur Sigma`가 높아(`{blur_sigma:.1f}`) 인접한 객체들의 경계가 뭉개져 하나로 합쳐질 수 있습니다."
        )
    elif noise_level > 0:
        feedbacks.append(
            f"**[노이즈 보정 작동 중]** 노이즈(`{noise_level}`)에 대해 스무딩(`{blur_sigma:.1f}`) 및 모폴로지(`{morph_size}`)가 적용되고 있습니다."
        )

    # 객체 수 및 크기 분포 진단
    if nums > 0:
        areas = cc[:, 1]
        median_area = float(np.median(areas))
        max_area = int(np.max(areas))
        if tiny_objects >= 5:
            feedbacks.append(
                f"**[미세 노이즈 객체 다수]** 30픽셀 미만의 극소 객체가 **{tiny_objects}개** 포함되어 전체 개수(`{nums}개`)가 부풀려졌습니다. "
                f"`최소 객체 픽셀 수`를 **{rec_min_area}** 이상으로 올리면 정확한 객체 수만 남습니다."
            )
        elif nums > 1 and max_area > median_area * 5:
            feedbacks.append(
                f"**[객체 병합 주의]** 평균 크기(`{median_area:.0f}px`)보다 훨씬 큰 덩어리(`{max_area}px`)가 존재합니다. "
                f"가까이 붙은 객체들이 하나로 합쳐졌을 가능성이 있으므로 채도 임계값을 조금 높여 확인해 보세요."
            )
        else:
            feedbacks.append(
                f"**[검출 안정]** 총 **{nums}개**의 객체가 비교적 균일한 크기(중앙값 `{median_area:.0f}px`)로 검출되었습니다."
            )

    for fb in feedbacks:
        st.markdown(f"- {fb}")

else:
    st.info("왼쪽 사이드 바에서 이미지 파일을 입력해주세요.")
