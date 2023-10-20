import cv2
import numpy as np
import streamlit as st

input_image = np.fromfile('img/hw3/actontBin.bin', dtype=np.uint8).reshape(256, 256)
img_copy = input_image.copy()
input_image_gray = cv2.cvtColor(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)

template = cv2.imread('img/hw3/tmp.jpg', cv2.IMREAD_GRAYSCALE)

result = cv2.matchTemplate(input_image_gray, template, cv2.TM_CCORR_NORMED)

J1 = result.copy()
J1 = cv2.normalize(J1, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

st.title("HW3.3")

col1, col2 = st.columns(2)
with col1:

    st.image(input_image, use_column_width=True, channels="L", caption="Original Image")
with col2:

    st.image(J1, use_column_width=True, caption="Construct output image J1")

threshold_value = 0.85

col3, col4 = st.columns(2)
with col3:
    J2 = np.where(J1 >= threshold_value, 255, 0).astype(np.uint8)
    st.image(J2, use_column_width=True, caption="J2 - Binary Image")

