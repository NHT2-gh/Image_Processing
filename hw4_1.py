import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2

I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)

Itilde1 = np.fft.fftshift(np.fft.fft2(I1))

print("Real part of I1:")
print(np.real(I1))
print("\nImaginary part of I1:")
print(np.imag(I1))

print("\nReal part of DFT(I1):")
np.set_printoptions(precision=4, suppress=True)
print(np.real(Itilde1))

print("\nImaginary part of DFT(I1):")
print(np.imag(Itilde1))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(np.real(I1), cmap='gray')
axes[0, 0].set_title('Real part of I1')
axes[0, 1].imshow(np.imag(I1), cmap='gray')
axes[0, 1].set_title('Imaginary part of I1')

axes[1, 0].imshow(np.real(Itilde1), cmap='gray')
axes[1, 0].set_title('Real part of DFT(I1)')
axes[1, 1].imshow(np.imag(Itilde1), cmap='gray')
axes[1, 1].set_title('Imaginary part of DFT(I1)')

axes[0, 2].axis('off')
axes[0, 3].axis('off')
axes[1, 2].axis('off')
axes[1, 3].axis('off')

axes[0, 2].text(0.1, 0.5, "Real(I1)\n\n" + str(np.real(I1)), fontsize=10)
axes[0, 3].text(0.1, 0.5, "Imag(I1)\n\n" + str(np.imag(I1)), fontsize=10)
axes[1, 2].text(0.1, 0.5, "Real(DFT(I1))\n\n" + str(np.real(Itilde1)), fontsize=10)
axes[1, 3].text(0.1, 0.5, "Imag(DFT(I1))\n\n" + str(np.imag(Itilde1)), fontsize=10)

plt.tight_layout()
plt.show()
