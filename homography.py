import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import os

def threshIt(img, thresh_min=10, thresh_max=160):
    """
    Applies a threshold to the `img` using [`thresh_min`, `thresh_max`] returning a binary image [0, 255]
    """
    xbinary = np.zeros_like(img)
    xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return xbinary


current_dir = os.path.dirname(os.path.abspath(__file__))
temp_path, dir_name = os.path.split(current_dir)
image_dir = os.path.join(current_dir, "data_1/data")

images = []
image_names = []

# sorted(os.listdir(whatever_directory))

K = np.asarray([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                [0.000000e+00, 9.019653e+02, 2.242509e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00]])

D = np.asarray([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

for name in sorted(os.listdir(image_dir)):
    # print(name)
    im = cv2.imread(os.path.join(image_dir, name))
    im = cv2.undistort(im, K, D) 
    im = cv2.GaussianBlur(im, (5,5), 20.0)
    if im is not None:
        images.append(im)
        image_names.append(name)
    else:
        print("None")

count = 0

for i in range(len(images)):

	image = images[i]
	img = deepcopy(image)

	# image = cv2.imread('data_1/data/0000000000.png')
	# img = deepcopy(image)

	#canny = cv2.Canny(gray, 50, 150)

	cv2.circle(img, (190, 500), 5, (255, 0, 0), -1)
	cv2.circle(img, (950, 500), 5, (255, 0, 0), -1)
	cv2.circle(img, (530, 300), 5, (255, 0, 0), -1)
	cv2.circle(img, (760, 300), 5, (255, 0, 0), -1)

	cv2.imshow("img_thresh", img)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()


	# src = np.float32([[450, 0], [450, 1100], [250, 0], [250, 1100]])
	# dst = np.float32([[223, 569], [223, 711], [0, 0], [0, 1280]])
	# H = cv2.getPerspectiveTransform(src, dst)

	src = np.float32([[190, 500], [950, 500], [530, 300], [760, 300]])
	dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
	Hom = cv2.getPerspectiveTransform(src, dst)

	warped_brg = cv2.warpPerspective(img, Hom, (200,400))
	# cv2.imshow("warped1", warped)
	# cv2.waitKey(0) 

	# img_thresh = combined_threshold(warped)
	# cv2.imshow("img_thresh", img_thresh)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# warped = cv2.GaussianBlur(warped, (1,1), 1.0)
	warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# warped = cv2.Canny(warped, 50, 150)
	sobel_x = cv2.Sobel(warped,cv2.CV_64F,1,0,ksize=3)
	sobel_y = cv2.Sobel(warped,cv2.CV_64F,0,1,ksize=3)

	sobel_x = np.absolute(sobel_x)
	sobel_x = np.uint8(255.0*sobel_x/np.max(sobel_x))
	sobel_x = threshIt(sobel_x, 25, 180)

	sobel_y = np.absolute(sobel_y)
	sobel_y = np.uint8(255.0*sobel_y/np.max(sobel_y))
	sobel_y = threshIt(sobel_y, 25, 150)

	grad = np.sqrt(sobel_x**2 + sobel_y**2)

	grad = ((255/np.max(grad)) * grad).astype(np.uint8) 

	grad = threshIt(grad, 180, 255)

	absgraddir = np.uint8(np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x)))
	absgraddir = threshIt(absgraddir, 1, np.pi/2)

	combined = np.zeros_like(sobel_y) 
	combined[((sobel_x == 1) & (sobel_y == 1)) | ((grad == 1) & (absgraddir == 1))] = 1

	# cv2.imshow("sobel_x", 255*sobel_x)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("sobel_y", 255*sobel_y)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("grad", 255*grad)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("absgraddir", 255*absgraddir)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("combined", 255*combined)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	H = hls[:,:,0]
	L = hls[:,:,1]
	S = hls[:,:,2]

	sbinary = threshIt(S, 40, 120)
	# cv2.imshow("H", H)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("L", L)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("S", S)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("sbinary", 255 * sbinary)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# RGB colour
	R = img[:,:,2]
	G = img[:,:,1]
	B = img[:,:,0]
	rbinary = threshIt(R, 170, 240)
	# cv2.imshow("R", R)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("warped2", G)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("warped2", B)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("rbinary", 255*rbinary)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# YUV colour
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	Y = yuv[:,:,0]
	U = yuv[:,:,1]
	V = yuv[:,:,2]
	ubinary = threshIt(Y, 180, 255)
	# cv2.imshow("U", U)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("Y", Y)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("V", V)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("ubinary", 255*ubinary)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# combined1 = np.zeros_like(warped)
	# combined1[(sobel_x == 1)  | (sbinary == 1) | (rbinary == 1)] = 1

	# combined2 = np.zeros_like(warped)
	# combined2[((sobel_x == 1) & (sobel_y == 1)) | (rbinary == 1)] = 1

	# combined3 = np.zeros_like(warped)
	# combined3[(sbinary ==1) | (ubinary ==1)| (rbinary == 1 )] = 1

	# combined4 = np.zeros_like(warped)
	# combined4[(sobel_x == 1)  | (sbinary == 1) | (rbinary == 1 ) ] = 1

	combined5 = np.zeros_like(warped)
	combined5[((sobel_x == 1) | (rbinary == 1)) & ( (sbinary ==1) | (ubinary ==1)| (rbinary == 1 ))] = 1

	# cv2.imshow("combined1", 255*combined1)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("combined1", 255*combined2)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("combined1", 255*combined3)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	cv2.imshow("combined1", 255*combined5)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()




	# combined1 = cv2.warpPerspective(combined1, Hom, (200,500))
	# combined2 = cv2.warpPerspective(combined2, Hom, (200,500))
	# combined3 = cv2.warpPerspective(combined3, Hom, (200,500))
	combined5 = cv2.warpPerspective(combined5, Hom, (200,500))

	# cv2.imshow("combined1-tran", 255*combined1)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("combined1-tran", 255*combined2)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("combined1-tran", 255*combined3)
	# if cv2.waitKey(0) & 0xff == 27:
	    # cv2.destroyAllWindows()
	cv2.imshow("combined1-tran", 255*combined5)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()


