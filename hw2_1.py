# Name: Nguyen Huynh Truc Thanh
# Id :  N20DCCN142

import numpy as np
import matplotlib.pyplot as plt

lena = np.fromfile('img/hw2/lena.bin', dtype=np.uint8).reshape(256, 256)
peppers = np.fromfile('img/hw2/peppers.bin', dtype=np.uint8).reshape(256, 256)

plt.figure()
plt.subplot(121)
plt.imshow(lena, cmap='gray')
plt.title('Lena Image')

plt.subplot(122)
plt.imshow(peppers, cmap='gray')
plt.title('Peppers Image')

plt.show()

J = np.zeros((256, 256), dtype=np.uint8)
J[:, :128] = lena[:, :128]
J[:, 128:] = peppers[:, 128:]

plt.figure()
plt.imshow(J, cmap='gray')
plt.title('Image J')

plt.show()

K = np.zeros((256, 256), dtype=np.uint8)
K[:, :128] = J[:, 128:]
K[:, 128:] = J[:, :128]

plt.figure()
plt.imshow(K, cmap='gray')
plt.title('Image K')

plt.show()