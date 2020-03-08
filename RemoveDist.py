import numpy as np
import cv2

img = cv.imread("data_1/data/0000000000.png")

K = np.asarray([[9.037596e+02, 0.000000e+00, 6.957519e+02],
				[0.000000e+00, 9.019653e+02, 2.242509e+02],
				[0.000000e+00, 0.000000e+00, 1.000000e+00]])

D = np.asarray([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

p, q, r = img.shape

for i in range(p):
	for j in range(q):
		cord = np.asarray([i, j, 1])
		i_new, j_new, scale = np.matmul(K, cord)
		img[int(i_new)][int(j_new)] = img[i][j]

