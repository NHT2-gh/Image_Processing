# Name: Nguyen Huynh Truc Thanh
# Id :  N20DCCN142

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage import exposure, io

st.title("Adaptive Histogram Equalization")
img = cv2.imread('img/dental.jpg', cv2.IMREAD_GRAYSCALE)

col1, col2 = st.columns(2)
col1.image(img, caption="Original Image", use_column_width=True)
col2.image(cv2.equalizeHist(img), caption="Original histogram equalization")

col3, col4 = st.columns(2)
col1.image(exposure.equalize_adapthist(img, kernel_size=(8,8)), caption="Adaptive histogram equalization, 8x8 titles", use_column_width=True)
col2.image(exposure.equalize_adapthist(img, kernel_size=(16,16)), caption="Adaptive histogram equalization, 16x16 titles",use_column_width=True)