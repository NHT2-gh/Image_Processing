
import cv2
import numpy as np
import streamlit as st
import matplotlib as plt


ori_image = np.fromfile('img/hw5/girl2.bin', dtype=np.uint8).reshape(256, 256)
noise32hi_image = np.fromfile('img/hw5/girl2Noise32Hi.bin', dtype=np.uint8).reshape(256, 256)
noise32_image = np.fromfile('img/hw5/girl2Noise32.bin ', dtype=np.uint8).reshape(256, 256)
# input_image_gray = cv2.cvtColor(cv2.cvtColor(noise32_image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)


# a.
def calculate_mse(image1, image2):
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    error = float("{: .4f}".format(error))
    return error


# Tính MSE
mseOri_ori = calculate_mse(ori_image, ori_image)
mseOri_Noise32 = calculate_mse(ori_image, noise32_image)
mseOri_NoiseHiPass = calculate_mse(ori_image, noise32hi_image)

# show result by streamlit
st.title("HW5.2")

tab1, tab2, tab3 = st.tabs(["2.A-B", "2.C", "2.c"])

with tab1:
    st.subheader("Display and calculate the MSE between the noisy image and the original image")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(ori_image, use_column_width=True, channels="L", caption="Original Image")
        st.write("MSE: ", mseOri_ori)
    with col2:
        st.image(noise32hi_image, use_column_width=True, channels="L", caption="High Pass White Gaussian Noise")
        st.write("MSE: ", mseOri_NoiseHiPass)

    with col3:
        st.image(noise32_image, use_column_width=True, channels="L", caption=" Broadband White Gaussian Noise")
        st.write("MSE: ", mseOri_Noise32)

# b.

#Tạo ma trận DFT 256x256
[U, V] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
U_cutoff = 64
HLtildeCenter = np.double(np.sqrt(U**2 + V**2) <= U_cutoff)
HLtilde = np.fft.fftshift(HLtildeCenter)


#Tính toán DFT của từng ảnh và áp dụng bộ lọc
def DFT_Filter(img_name):
    dft = np.fft.fft2(img_name)
    filtered_img = np.fft.ifft2(dft * HLtilde).real
    cv2.imwrite('img/hw5/tmp.jpg', filtered_img)
    return filtered_img
st.subheader("Displays and calculates MSE between filtered and pre-filtered images")
col4, col5, col6 = st.columns(3)

DFT_Filter(ori_image)
with col4:
    st.image("img/hw5/tmp.jpg", use_column_width=True, channels="L", clamp=True,caption="Filtered girl2")
DFT_Filter(noise32_image)
with col6:
    st.image("img/hw5/tmp.jpg", use_column_width=True, channels="L", clamp=True,caption="Filtered girl2Noise32")

DFT_Filter(noise32hi_image)
with col5:
    st.image("img/hw5/tmp.jpg", use_column_width=True, channels="L", clamp=True,caption="Filtered girl2Noise32Hi")

# Tính toán MSE
def mse_filter(filter_img):
    mse_filter = np.mean((DFT_Filter(filter_img) - filter_img)**2)
    mse_filter = float("{: .4f}".format(mse_filter))
    return mse_filter

with col4:
    st.write('MSE Of Filtered Image', mse_filter(ori_image))
with col5:
    st.write('MSE Of Filtered Image', mse_filter(noise32_image))
with col6:
    st.write('MSE Of Filtered Image', mse_filter(noise32hi_image))

# c.

# Định nghĩa tần số cắt và hằng số không gian
U_cutoff_H = 64
SigmaH = 0.19 * 256 / U_cutoff_H

# Tạo lưới cho các giá trị tần số không gian
[U, V] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))

# Tính toán HtildeCenter
HtildeCenter = np.exp((-2 * np.pi**2 * SigmaH**2) / (256**2) * (U**2 + V**2))

# Sử dụng fftshift để "giữa" mảng DFT
Htilde = np.fft.fftshift(HtildeCenter)

# Áp dụng fftshift lại để "trung tâm" hồi đáp đimpulse
H = np.fft.ifft2(Htilde)
H2 = np.fft.fftshift(H)

# Tạo một ma trận zero với kích thước 512x512
ZPH2 = np.zeros((512, 512))
ZPH2[:256, :256] = H2

# Lấy DFT 512x512 của ZPH2 và nhân nó điểm với DFT của ảnh zero padded để
# thu được DFT của ba ảnh đã lọc zero padded
# (Phần này có thể thực hiện trên ảnh gốc hoặc ảnh có nhiễu, tùy thuộc vào ứng dụng của bạn)

# Hiển thị và tính toán MSE và ISNR cho mỗi ảnh đã lọc
# (Giống như phần trước, chắc chắn sử dụng ảnh dạng float từ DFT nghịch đảo mà không làm tròn và không áp dụng bất kỳ điều chỉnh tỷ lệ cự li đầy đủ)

DFT_ZPH2 = np.fft.ifft2(ZPH2)

def zeroPadding_DFT(img):
    img_padded = np.pad(img, ((3, 3), (3, 3)), mode='constant', constant_values=0)



