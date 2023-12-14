import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to read binary file
def read_bin(file_path, size):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8, count=size*size)
        return np.reshape(data, (size, size))

# Function to stretch image values
def stretch(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0

# Read the binary file
file_path = ('img/hw5/salesman.bin')
size = 256
X = read_bin(file_path, size)

# Display the original image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Original Image', fontsize=18)


# Apply 7x7 average filter using linear convolution
X2 = np.zeros((262, 262))
X2[4:260, 4:260] = X
Y2 = np.zeros((262, 262))
for row in range(4, 261):
    for col in range(4, 261):
        Y2[row, col] = np.sum(X2[row-3:row+4, col-3:col+4]) / 49

# Stretch the values and crop the result
Y = stretch(Y2[4:260, 4:260])

# Display the filtered image
plt.subplot(1,2, 2)
plt.imshow(Y, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Filtered Image', fontsize=18)

plt.show()

# Save the result image for comparison with later results
Y1a = Y


#---------------------------b-----------------------



# Make the 128x128 impulse response image

plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(X, cmap='gray')
plt.title('Original image', fontsize=12)
plt.axis('image')
plt.axis('off')


# Zero pad the original image and the H image
Padsize = 256 + 128 - 1
ZPX = np.zeros((Padsize, Padsize))
ZPX[:256, :256] = X



plt.subplot(2, 4, 2)
plt.imshow(ZPX, cmap='gray')
plt.title('Zero Padded', fontsize=12)
plt.axis('image')
plt.axis('off')



H = np.zeros((128, 128))
H[62:69, 62:69] = 1 / 49

ZPH = np.zeros((Padsize, Padsize))
ZPH[:128, :128] = H

# Display zero-padded impulse response
plt.subplot(2, 4, 3)
plt.imshow(ZPH, cmap='gray')
plt.title('Zero Padded Impulse Resp', fontsize=12)
plt.axis('image')
plt.axis('off')


# Compute DFT's of zero-padded images
ZPXtilde = np.fft.fft2(ZPX)
ZPHtilde = np.fft.fft2(ZPH)

# Show centered log-magnitude spectra
ZPXtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(ZPXtilde)))

plt.subplot(2, 4, 4)
plt.imshow(ZPXtildeDisplay, cmap='gray')
plt.title('Log-mag spectrum zero pad', fontsize=12)
plt.axis('image')
plt.axis('off')


ZPHtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(ZPHtilde)))
plt.subplot(2, 4, 5)
plt.imshow(ZPHtildeDisplay, cmap='gray')
plt.title('Log-magnitude spectrum H', fontsize=12)
plt.axis('image')
plt.axis('off')


# Compute the convolution by pointwise multiplication of DFT's
ZPYtilde = ZPXtilde * ZPHtilde
ZPY = np.fft.ifft2(ZPYtilde)

# Show the resulting zero-padded image and its centered log-magnitude spectrum
ZPYtildeDisplay = np.log(1 + np.abs(np.fft.fftshift(ZPYtilde)))
plt.subplot(2, 4, 6)
plt.imshow(ZPYtildeDisplay, cmap='gray')
plt.title('Log-magnitude spectrum of result', fontsize=12)
plt.axis('image')
plt.axis('off')


plt.subplot(2, 4, 7)
plt.imshow(np.real(ZPY), cmap='gray')
plt.title('Zero Padded Result', fontsize=12)
plt.axis('image')
plt.axis('off')


# Extract the final result image and display
Y = np.real(ZPY[64:320, 64:320])
plt.subplot(2, 4, 8)
plt.imshow(Y, cmap='gray')
plt.title('Final Filtered Image', fontsize=12)
plt.axis('image')
plt.axis('off')
plt.show()


#-------------- c---------------------

# Function to perform full-scale contrast stretch
def stretch(x):
    xMax = np.max(x)
    xMin = np.min(x)
    scale_factor = 255.0 / (xMax - xMin)
    y = np.round((x - xMin) * scale_factor)
    return y.astype(np.uint8)

# Load the original image


# Make the 256x256 impulse response image
H1 = np.zeros((256, 256))
H1[126:133, 126:133] = 1/49

# Get the true zero-phase impulse response image using fftshift
H2 = np.fft.fftshift(H1)


# Display the zero-phase impulse response image
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(stretch(X), cmap='gray')
plt.title('Zero Phase Impulse Resp', fontsize=18)
plt.axis('image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(stretch(H2), cmap='gray')
plt.title('Zero Phase Impulse Resp', fontsize=18)
plt.axis('image')
plt.axis('off')

# Zero pad the input image
ZPX = np.zeros((512, 512))
ZPX[:256, :256] = X

# Make the zero-padded zero-phase impulse response image
ZPH2 = np.zeros((512, 512))
ZPH2[:128, :128] = H2[:128, :128]
ZPH2[:128, 385:512] = H2[:128, 129:256]
ZPH2[385:512, :128] = H2[129:256, :128]
ZPH2[385:512, 385:512] = H2[129:256, 129:256]

# Display the zero-padded zero-phase impulse response image
plt.subplot(2, 2, 3)
plt.imshow(stretch(ZPH2), cmap='gray')
plt.title('Zero Padded zero-phase H', fontsize=18)
plt.axis('image')
plt.axis('off')


# Compute the filtered result by pointwise multiplication of DFTs
Y = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPH2))
Y = stretch(Y[:256, :256])

# Display the final filtered image
plt.subplot(2, 2, 4)
plt.imshow(Y, cmap='gray')
plt.title('Final Filtered Image', fontsize=18)
plt.axis('image')
plt.axis('off')
plt.show()







