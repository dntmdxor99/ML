import numpy as np
import cv2

img = cv2.imread('Python/study/test4.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (400, 600))

new_x = np.zeros(img.shape)
new_y = np.zeros(img.shape)

img = np.pad(img, (1, 1), 'constant', constant_values = 0)

mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
mask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

for i in range(1, img.shape[0] - mask_x.shape[0] + 2):
    for j in range(1, img.shape[1] - mask_y.shape[0] + 2):
        new_x[i - 1, j - 1] = np.sum(img[i - 1 : i + 2, j - 1 :j + 2] * mask_x)
        new_y[i - 1, j - 1] = np.sum(img[i - 1 : i + 2, j - 1 : j + 2] * mask_y)

new_xy = np.add(new_x, new_y)
new_x = (new_x - np.min(new_x)) / np.ptp(new_x)
new_y = (new_y - np.min(new_y)) / np.ptp(new_y)
new_xy = (new_xy - np.min(new_xy)) / np.ptp(new_xy)

cv2.imshow('original', img)
cv2.imshow('derivative_x', new_x)
cv2.imshow('derivative_y', new_y)
cv2.imshow('derivative_xy', new_xy)
cv2.waitKey()
cv2.destroyAllWindows()