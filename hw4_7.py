import numpy as np
import matplotlib.pyplot as plt

I6 = np.fromfile('img/hw4/camera.bin', dtype=np.uint8).reshape(256, 256)

J1 = np.abs(I6)
J2 = np.angle(I6)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].set_title('J2: DFT Phase Contribution')
axes[0].imshow(J2, cmap='gray', vmin=-np.pi, vmax=np.pi)
axes[0].axis('off')

JJ1 = np.log(J1 + 1e-10)

axes[1].set_title('JJ1: Log(DFT Magnitude Contribution)')
axes[1].imshow(JJ1, cmap='gray')
axes[1].axis('off')

plt.show()
