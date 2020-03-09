import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import os


# def sharpen_img(img):
#     gb = cv2.GaussianBlur(img, (5,5), 20.0)
#     return cv2.addWeighted(img, 2, gb, -1, 0)

# # Compute linear image transformation img*s+m
# def lin_img(img,s=1.0,m=0.0):
#     img2=cv2.multiply(img, np.array([s]))
#     return cv2.add(img2, np.array([m]))

# # Change image contrast; s>1 - increase
# def contr_img(img, s=1.0):
#     m=127.0*(1.0-s)
#     return lin_img(img, s, m)


# def binary_filter_road_pavement(img):
    
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     # create dimensions for roi boxes - center, and left & right half height
#     width=img.shape[1]
#     height=img.shape[0]
#     x_center=np.int(width/2)
#     roi_width=100
#     roi_height=200
#     x_left=x_center-np.int(roi_width/2)
#     x_right=x_center+np.int(roi_width/2)
#     y_top=height-30
#     y_bottom=y_top-roi_height
#     y_bottom_small=y_top-np.int(roi_height/2)
#     x_offset=50
#     x_finish=width-x_offset
    
#     # extract the roi and stack before converting to HSV
#     roi_center=img[y_bottom:y_top, x_left:x_right]
#     roi_left=img[y_bottom_small:y_top, x_offset:roi_width+x_offset]
#     roi_right=img[y_bottom_small:y_top, x_finish-roi_width:x_finish]
#     roi=np.hstack((roi_center,np.vstack((roi_left,roi_right))))
#     roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
#         # calculating object histogram
#     roihist = cv2.calcHist([roi_hsv],[0, 1], None, [256, 256], [0, 256, 0, 256] )
# #     roihist = cv2.calcHist([roi_hsv],[0], None, [256], [0, 256] )
    
#     # normalize histogram and apply backprojection
#     cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
#     dst = cv2.calcBackProject([img_hsv],[0,1],roihist,[0,256,0,256],1)
# #     dst = cv2.calcBackProject([img_hsv],[0],roihist,[0,256],1)

#     # Now convolute with circular disc
#     disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#     cv2.filter2D(dst,-1,disc,dst)
    
#     # threshold and binary AND
#     ret,thresh = cv2.threshold(dst,10,250,cv2.THRESH_BINARY_INV)

#     return thresh

# def combined_threshold(img, kernel=3, grad_thresh=(30,100), mag_thresh=(70,100), dir_thresh=(0.8, 0.9),
#                        s_thresh=(100,255), r_thresh=(150,255), u_thresh=(140,180),
#                       # threshold="daytime-normal"):
#     threshold="daytime-filter-pavement"):

#     def binary_thresh(channel, thresh = (200, 255), on = 1):
#         binary = np.zeros_like(channel)
#         binary[(channel > thresh[0]) & (channel <= thresh[1])] = on

#         return binary
    
#     # overwrite defaults
# #     if threshold == "daytime-shadow":
# #         grad_thresh=(15,100)

#     # check up the default red_min threshold to cut out noise and detect white lines
#     if threshold in ["daytime-bright","daytime-filter-pavement"]:
#          r_thresh=(210,255)
        
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Take both Sobel x and y gradients
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    
#     # calculate the scobel x gradient binary
#     abs_sobelx = np.absolute(sobelx)
#     scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
#     gradx = binary_thresh(scaled_sobelx, grad_thresh)
    
#     # calculate the scobel y gradient binary
#     abs_sobely = np.absolute(sobely)
#     scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
#     grady = binary_thresh(scaled_sobely, grad_thresh)
    
#     # Calculate the gradient magnitude
#     gradmag = np.sqrt(sobelx**2 + sobely**2)
#     # Rescale to 8 bit
#     scale_factor = np.max(gradmag)/255 
#     gradmag = (gradmag/scale_factor).astype(np.uint8) 
#     mag_binary = binary_thresh(gradmag, mag_thresh)
#     cv2.imshow("mag_binary", mag_binary)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
    
#     # Take the absolute value of the gradient direction, 
#     # apply a threshold, and create a binary image result
#     absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
#     dir_binary = binary_thresh(absgraddir, dir_thresh)
#     cv2.imshow("dir_binary", dir_binary)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
    
#     # HLS colour
#     hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#     H = hls[:,:,0]
#     L = hls[:,:,1]
#     S = hls[:,:,2]
#     sbinary = binary_thresh(S, s_thresh)
#     cv2.imshow("sbinary", sbinary)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
    
#     # RGB colour
#     R = img[:,:,2]
#     G = img[:,:,1]
#     B = img[:,:,0]
#     rbinary = binary_thresh(R, r_thresh)
#     cv2.imshow("rbinary", rbinary)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
    
#     # YUV colour
#     yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     Y = yuv[:,:,0]
#     U = yuv[:,:,1]
#     V = yuv[:,:,2]
#     ubinary = binary_thresh(U, u_thresh)
#     cv2.imshow("ubinary", ubinary)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
    
#     combined = np.zeros_like(dir_binary)
    
#     if threshold == "daytime-normal": # default
#         combined[(gradx == 1)  | (sbinary == 1) | (rbinary == 1) ] = 1
#     elif threshold == "daytime-shadow":
#         combined[((gradx == 1) & (grady == 1)) | (rbinary == 1)] = 1
#     elif threshold == "daytime-bright":
#         combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sbinary ==1 ) | (rbinary == 1)] = 1
#     elif threshold == "daytime-filter-pavement":
#         road_binary=binary_thresh(binary_filter_road_pavement(img))
#         combined[(((gradx == 1)  | (sbinary == 1)) & (road_binary==1)) | (rbinary == 1 ) ] = 1
#     else:
#         combined[((gradx == 1) | (rbinary == 1)) & ( (sbinary ==1) | (ubinary ==1)| (rbinary == 1 ))] = 1

#     cv2.imshow("warped2", combined)
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
        
#     return combined

def threshIt(img, thresh_min=10, thresh_max=160):
    """
    Applies a threshold to the `img` using [`thresh_min`, `thresh_max`] returning a binary image [0, 255]
    """
    xbinary = np.zeros_like(img)
    xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return xbinary

def binary_thresh(channel, thresh = (200, 255), on = 1):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = on

    return binary


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
	# print(np.max(absgraddir))
	absgraddir = threshIt(absgraddir, 1, np.pi/2)

	# sobel_x = np.absolute(sobel_x)
	# sobel_x = np.uint8(255.0*sobel_x/np.max(sobel_x))
	# sobel_x = threshIt(sobel_x, 20, 160)

	# sobel_y = np.absolute(sobel_y)
	# sobel_y = np.uint8(255.0*sobel_y/np.max(sobel_y))
	# sobel_y = threshIt(sobel_y, 5, 160)

	combined = np.zeros_like(sobel_y) 
	combined[((sobel_x == 1) & (sobel_y == 1)) | ((grad == 1) & (absgraddir == 1))] = 1

	# xbinary = np.zeros_like(img)

	# p, q = sobel_x.shape
	# for i in range(p):
	#     for j in range(q):
	#         if grad[i][j] > 45:
	#             grad[i][j] = 255
	#         else:
	#             grad[i][j] = 0;
	# # grad = threshIt(grad, 10, 50)



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

	sbinary = binary_thresh(S, (40,120))
	# sbinary = cv2.warpPerspective(sbinary, Hom, (200,500))
	th2 = cv2.adaptiveThreshold(S,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(S,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
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
	# # cv2.imshow("th2", th2)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("th3", th3)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# RGB colour
	R = img[:,:,2]
	G = img[:,:,1]
	B = img[:,:,0]
	# R = warped_brg[:,:,2]
	# G = warped_brg[:,:,1]
	# B = warped_brg[:,:,0]
	rbinary = (binary_thresh(R, (170,240)))
	# rbinary = cv2.warpPerspective(rbinary, Hom, (200,500))
	# cv2.imshow("R", R)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("warped2", warped_brg)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("rbinary", 255*rbinary)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("warped2", G)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()
	# cv2.imshow("warped2", B)
	# if cv2.waitKey(0) & 0xff == 27:
	#     cv2.destroyAllWindows()

	# YUV colour
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	Y = yuv[:,:,0]
	U = yuv[:,:,1]
	V = yuv[:,:,2]
	ubinary = binary_thresh(Y, (180,255))
	# ubinary = cv2.warpPerspective(ubinary, Hom, (200,500))
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


