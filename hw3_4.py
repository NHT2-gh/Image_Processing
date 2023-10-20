import streamlit as st
import numpy as np
from PIL import Image
import os

with open("img/hw3/johnny.bin", "rb") as f:
    image_data = np.fromfile(f, dtype=np.uint8, count=256 * 256)

image_data = image_data.reshape(256, 256)

histogram = np.histogram(image_data, bins=256, range=(0, 256))[0]

cdf = np.cumsum(histogram)

cdf_normalized = cdf * 255 / cdf[-1]

equalized_image_data = cdf_normalized[image_data]

equalized_image_data = equalized_image_data.astype(np.uint8)

equalized_image = Image.fromarray(equalized_image_data)

equalized_histogram = np.histogram(equalized_image_data, bins=256, range=(0, 256))[0]

st.title("HW3.4")

col1, col2 = st.columns(2)

with col1:

    st.image(image_data, use_column_width=True, channels="L", caption="Original Image")
with col2:

    st.image(equalized_image_data, use_column_width=True, caption="Equalized Image")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Original Image")
    st.bar_chart(histogram)
with col4:
    st.subheader("Histogram Equalized Image")
    st.bar_chart(equalized_histogram)