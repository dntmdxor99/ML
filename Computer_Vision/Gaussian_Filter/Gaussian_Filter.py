import cv2
import numpy as np


def gaussian_distribution(sigma = 1) -> np.array:
    x, y = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5))
    
    z = np.exp(-((np.square(x) / (2 * np.square(sigma)) + (np.square(y) / (2 * np.square(sigma)))))) / (2 * np.pi * sigma * sigma)
    z = np.sum(z, axis = 1) / 2
    
    return z


img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (500, 700), interpolation=cv2.INTER_LANCZOS4)

gaussian = gaussian_distribution(10)
g_half = gaussian.shape[0] // 2

img_padded = np.pad(img, (g_half, g_half), 'constant', constant_values = 0)

filtered_img = np.zeros(img.shape)

for i in range(g_half, img_padded.shape[0] - g_half):     # 세로
    for j in range(g_half, img_padded.shape[1] - g_half):     # 가로
        filtered_img[i - g_half, j - g_half] = np.sum(gaussian * img_padded[i, j - g_half : j + g_half + 1])
        filtered_img[i - g_half, j - g_half] += np.sum(gaussian * img_padded[i - g_half : i + g_half + 1, j])

filtered_img = ((filtered_img - np.min(filtered_img)) / np.ptp(filtered_img)) * 255
filtered_img = filtered_img.astype(np.uint8)

cv2.imshow('original', img)
cv2.imshow('gaussian filtering', filtered_img)
cv2.waitKey()
cv2.destroyAllWindows()