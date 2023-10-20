# Name: Nguyen Huynh Truc Thanh
# Id :  N20DCCN142
import numpy as np
import streamlit as st

def threshold_image(grayscale_image, threshold):
    binary_image = (grayscale_image > threshold) * 255
    return binary_image

# b
def generate_contour_image(binary_image):
    contour_image = np.zeros_like(binary_image)
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            if binary_image[i, j] == 0:
                continue
            neighbors = [binary_image[i - 1, j - 1], binary_image[i - 1, j], binary_image[i - 1, j + 1],
                         binary_image[i, j - 1], binary_image[i, j + 1],
                         binary_image[i + 1, j - 1], binary_image[i + 1, j], binary_image[i + 1, j + 1]]
            if 0 in neighbors:
                contour_image[i, j] = 255
    return contour_image

st.title("HW3.1")

with open("img/hw3/Mammogram_256.bin", "rb") as f:
    image_data = np.fromfile(f, dtype=np.uint8, count=256 * 256)

grayscale_image = image_data.reshape(256, 256)

threshold_value = 108
binary_image = threshold_image(grayscale_image, threshold_value)
contour_image = generate_contour_image(binary_image)

col1, col2 = st.columns(2)
with col1:
    st.image(binary_image, caption="Binary Image", use_column_width=True)
with col2:
    st.image(contour_image, caption="Contour Image", use_column_width=True)