import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2

I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I2 = np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I4 = -1j * (I1 - I2)

Itilde4 = np.fft.fftshift(np.fft.fft2(I4))

print("Real part of I4:")
print(np.real(I4))
print("\nImaginary part of I4:")
print(np.imag(I4))

print("\nReal part of DFT(I4):")
np.set_printoptions(precision=4, suppress=True)
print(np.real(Itilde4))

print("\nImaginary part of DFT(I4):")
print(np.imag(Itilde4))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(np.real(I4), cmap='gray')
axes[0, 0].set_title('Real part of I4')
axes[0, 1].imshow(np.imag(I4), cmap='gray')
axes[0, 1].set_title('Imaginary part of I4')

axes[1, 0].imshow(np.real(Itilde4), cmap='gray')
axes[1, 0].set_title('Real part of DFT(I4)')
axes[1, 1].imshow(np.imag(Itilde4), cmap='gray')
axes[1, 1].set_title('Imaginary part of DFT(I4)')

axes[0, 2].axis('off')
axes[0, 3].axis('off')
axes[1, 2].axis('off')
axes[1, 3].axis('off')

axes[0, 2].text(0.1, 0.5, "Real(I4)\n\n" + str(np.real(I4)), fontsize=10)
axes[0, 3].text(0.1, 0.5, "Imag(I4)\n\n" + str(np.imag(I4)), fontsize=10)
axes[1, 2].text(0.1, 0.5, "Real(DFT(I4))\n\n" + str(np.real(Itilde4)), fontsize=10)
axes[1, 3].text(0.1, 0.5, "Imag(DFT(I4))\n\n" + str(np.imag(Itilde4)), fontsize=10)

plt.tight_layout()
plt.show()
