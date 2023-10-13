# Name: Nguyen Huynh Truc Thanh
# Id :  N20DCCN142

import cv2
import numpy as np
import streamlit as st

def ContrastLimit(input_image, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image_output = clahe.apply(input_image)
    return image_output

st.title("Contrast Limit Histogram Equalization")

image_path = "img/moon.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrastLimit_output = ContrastLimit(image)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_column_width=True)
    col2.image(contrastLimit_output, caption="Contrast-limited histogram equalization")

    hist_orig, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist_adaptive, _ = np.histogram(contrastLimit_output.ravel(), bins=256, range=(0, 256))

    col3, col4 = st.columns(2)
    col3.write("Histograms of the original image:")
    col3.line_chart(hist_orig)
    col4.write("Contrast-limited histogram equalization: ")
    col4.line_chart(hist_adaptive)
else:
    st.write(f"Image file '{image_path}' does not exist.")