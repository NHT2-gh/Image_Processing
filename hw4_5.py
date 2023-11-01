import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
u1, v1 = 1.5, 1.5

I5 = np.cos(2 * np.pi * (u1 * COLS + v1 * ROWS))

Itilde5 = np.fft.fftshift(np.fft.fft2(I5))

print("Real part of I5:")
print(np.real(I5))
print("\nImaginary part of I5:")
print(np.imag(I5))

print("\nReal part of DFT(I5):")
np.set_printoptions(precision=4, suppress=True)
print(np.real(Itilde5))

print("\nImaginary part of DFT(I5):")
print(np.imag(Itilde5))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(np.real(I5), cmap='gray')
axes[0, 0].set_title('Real part of I5')
axes[0, 1].imshow(np.imag(I5), cmap='gray')
axes[0, 1].set_title('Imaginary part of I5')

axes[1, 0].imshow(np.real(Itilde5), cmap='gray')
axes[1, 0].set_title('Real part of DFT(I5)')
axes[1, 1].imshow(np.imag(Itilde5), cmap='gray')
axes[1, 1].set_title('Imaginary part of DFT(I5)')

axes[0, 2].axis('off')
axes[0, 3].axis('off')
axes[1, 2].axis('off')
axes[1, 3].axis('off')

axes[0, 2].text(0.1, 0.5, "Real(I5)\n\n" + str(np.real(I5)), fontsize=10)
axes[0, 3].text(0.1, 0.5, "Imag(I5)\n\n" + str(np.imag(I5)), fontsize=10)
axes[1, 2].text(0.1, 0.5, "Real(DFT(I5))\n\n" + str(np.real(Itilde5)), fontsize=10)
axes[1, 3].text(0.1, 0.5, "Imag(DFT(I5))\n\n" + str(np.imag(Itilde5)), fontsize=10)

plt.tight_layout()
plt.show()
