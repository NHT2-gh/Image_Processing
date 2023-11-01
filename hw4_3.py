import numpy as np
import matplotlib.pyplot as plt

COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

u0 = 2
v0 = 2

I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I2 = np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)
I3 = I1 + I2

Itilde3 = np.fft.fftshift(np.fft.fft2(I3))

print("Real part of I3:")
print(np.real(I3))
print("\nImaginary part of I3:")
print(np.imag(I3))

print("\nReal part of DFT(I3):")
np.set_printoptions(precision=4, suppress=True)
print(np.real(Itilde3))

print("\nImaginary part of DFT(I3):")
print(np.imag(Itilde3))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(np.real(I3), cmap='gray')
axes[0, 0].set_title('Real part of I3')
axes[0, 1].imshow(np.imag(I3), cmap='gray')
axes[0, 1].set_title('Imaginary part of I3')

axes[1, 0].imshow(np.real(Itilde3), cmap='gray')
axes[1, 0].set_title('Real part of DFT(I3)')
axes[1, 1].imshow(np.imag(Itilde3), cmap='gray')
axes[1, 1].set_title('Imaginary part of DFT(I3)')

axes[0, 2].axis('off')
axes[0, 3].axis('off')
axes[1, 2].axis('off')
axes[1, 3].axis('off')

axes[0, 2].text(0.1, 0.5, "Real(I3)\n\n" + str(np.real(I3)), fontsize=10)
axes[0, 3].text(0.1, 0.5, "Imag(I3)\n\n" + str(np.imag(I3)), fontsize=10)
axes[1, 2].text(0.1, 0.5, "Real(DFT(I3))\n\n" + str(np.real(Itilde3)), fontsize=10)
axes[1, 3].text(0.1, 0.5, "Imag(DFT(I3))\n\n" + str(np.imag(Itilde3)), fontsize=10)

plt.tight_layout()
plt.show()
