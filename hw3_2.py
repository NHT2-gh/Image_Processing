import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

bin_file_path = 'img/hw3/lady.bin'

with open(bin_file_path, 'rb') as file:
    image = np.fromfile(file, dtype=np.uint8).reshape(256, 256)

st.title('Histogram Stretching')

min_val = np.min(image)
max_val = np.max(image)
stretched_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

st.title("HW3.2")

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption='Original Image', use_column_width=True, channels='GRAY')
with col2:
    st.image(stretched_image, caption='Stretched Image', use_column_width=True, channels='GRAY')

col3, col4 = st.columns(2)
with col3:
    fig_original = plt.figure(figsize=(6, 6))
    plt.hist(image.ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.7)
    plt.title('Original Image Histogram')
    st.pyplot(fig_original)

with col4:
    fig_stretched = plt.figure(figsize=(6, 6))
    plt.hist(stretched_image.ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.7)
    plt.title('Stretched Image')
    st.pyplot(fig_stretched)