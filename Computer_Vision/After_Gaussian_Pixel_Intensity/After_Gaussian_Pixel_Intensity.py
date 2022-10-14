import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (500, 700))


fig, ax = plt.subplots(2, 3, figsize = (9, 9))
k = 0

for i in range(0, 2):
    for j in range(0, 3):
        ax[i, j].set(xlim = [0, 500], ylim = [0, 255])
        ax[i, j].plot(np.arange(0, img.shape[1]), img[250, :], color = 'b')
        ax[i, j].set_title(f'Gaussian_{k}')
        k += 1
        img = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX = 5, sigmaY = 5)


plt.show()