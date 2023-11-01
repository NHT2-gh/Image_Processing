import numpy as np
import matplotlib.pyplot as plt

image_filenames = ['img/hw4/camera.bin', 'img/hw4/salesman.bin', 'img/hw4/head.bin', 'img/hw4/eyeR.bin']

def process_and_display_image(image_filename):

    image_data = np.fromfile(image_filename, dtype=np.uint8).reshape(256, 256)

    image_dft = np.fft.fftshift(np.fft.fft2(image_data))

    real_part = np.real(image_dft)
    imaginary_part = np.imag(image_dft)

    magnitude_spectrum = np.log(np.abs(image_dft) + 1)

    phase = np.angle(image_dft)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].set_title('Original Image')
    axes[0].imshow(image_data, cmap='gray', vmin=0, vmax=255)
    axes[0].axis('off')

    axes[1].set_title('Real Part DFT')
    axes[1].imshow(real_part, cmap='gray', vmin=-1, vmax=1)
    axes[1].axis('off')

    axes[2].set_title('Imaginary Part DFT')
    axes[2].imshow(imaginary_part, cmap='gray', vmin=-1, vmax=1)
    axes[2].axis('off')

    axes[3].set_title('Log-Magnitude Spectrum')
    axes[3].imshow(magnitude_spectrum, cmap='gray')
    axes[3].axis('off')

    axes[4].set_title('Phase DFT')
    axes[4].imshow(phase, cmap='gray', vmin=-np.pi, vmax=np.pi)
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

for image_filename in image_filenames:
    process_and_display_image(image_filename)
