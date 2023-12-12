import cv2
import numpy as np
import streamlit as st

# a.

input_image = np.fromfile('img/hw5/salesman.bin', dtype=np.uint8).reshape(256, 256)

kernel = np.ones((7,7), np.float32)/ 49
image_padding = np.pad(input_image, ((3,3), (3,3)), mode='constant',constant_values=0)
ROI = image_padding[4:260, 4:260]
result = cv2.filter2D(ROI, -1, kernel)
image_padding[4:260, 4:260] = result

st.title("HW5.1")
col1, col2 = st.columns(2)
with col1:
    st.image(input_image, use_column_width=True, channels="L", caption="Original Image")
with col2:
    st.image(result, use_column_width=True, channels="L", caption="Filter Image")

# b.

