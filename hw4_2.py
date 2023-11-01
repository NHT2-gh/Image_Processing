import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2

I2 = np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)

Itilde2 = np.fft.fftshift(np.fft.fft2(I2))

print("Real part of I2:")
print(np.real(I2))
print("\nImaginary part of I2:")
print(np.imag(I2))

print("\nReal part of DFT(I2):")
np.set_printoptions(precision=4, suppress=True)
print(np.real(Itilde2))

print("\nImaginary part of DFT(I2):")
print(np.imag(Itilde2))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(np.real(I2), cmap='gray')
axes[0, 0].set_title('Real part of I2')
axes[0, 1].imshow(np.imag(I2), cmap='gray')
axes[0, 1].set_title('Imaginary part of I2')

axes[1, 0].imshow(np.real(Itilde2), cmap='gray')
axes[1, 0].set_title('Real part of DFT(I2)')
axes[1, 1].imshow(np.imag(Itilde2), cmap='gray')
axes[1, 1].set_title('Imaginary part of DFT(I2)')

axes[0, 2].axis('off')
axes[0, 3].axis('off')
axes[1, 2].axis('off')
axes[1, 3].axis('off')

axes[0, 2].text(0.1, 0.5, "Real(I2)\n\n" + str(np.real(I2)), fontsize=10)
axes[0, 3].text(0.1, 0.5, "Imag(I2)\n\n" + str(np.imag(I2)), fontsize=10)
axes[1, 2].text(0.1, 0.5, "Real(DFT(I2))\n\n" + str(np.real(Itilde2)), fontsize=10)
axes[1, 3].text(0.1, 0.5, "Imag(DFT(I2))\n\n" + str(np.imag(Itilde2)), fontsize=10)

plt.tight_layout()
plt.show()
