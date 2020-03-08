import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt


def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

# Compute linear image transformation img*s+m
def lin_img(img,s=1.0,m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

# Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m=127.0*(1.0-s)
    return lin_img(img, s, m)


image = cv2.imread('/home/akanksha/Downloads/Problem 2/data_1/data/0000000000.png')
img = deepcopy(image)


height, width, channels = img.shape
print(height)
print(width)
print(channels)
#canny = cv2.Canny(gray, 50, 150)


src = np.float32([[0, 450], [1100, 450], [0, 250], [1100, 250]])
dst = np.float32([[569, 223], [711, 223], [0, 0], [1280, 0]])
H = cv2.getPerspectiveTransform(src, dst)

warped = cv2.warpPerspective(img, H, (width,height))
warped = sharpen_img(warped)
warped = contr_img(warped, 1.1)

plt.plot()
plt.imshow(img)
plt.savefig('test')

plt.close()

plt.imshow(warped)
plt.savefig('warped')

plt.close()
#cv2.imshow("result", img)
#if cv2.waitKey(0) & 0xff == 27:
#	cv2.destroyAllWindows()
