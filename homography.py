import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import math

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

select_data = 1

K = np.asarray([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                [0.000000e+00, 9.019653e+02, 2.242509e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00]])

D = np.asarray([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

if select_data == 0:
    cap = cv2.VideoCapture('data_2/challenge_video.mp4')
    if (cap.isOpened() == False):
        print("Unable to read camera feed")
    # out = cv2.VideoWriter('tag1CubeResult.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))
    while (True):
        ret, im = cap.read()
        if ret == True:
            im = cv2.undistort(im, K, D)
            img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            im = cv2.GaussianBlur(im, (5,5), 5)
            images.append(im)
        else:
            cap.release()
            break

    src1 = np.array([[(280, 700), (1100, 700), (760, 480), (600, 480)]])

    mask = np.zeros_like(images[0])
    cv2.fillPoly(mask, src1, (255,255,255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = threshIt(mask, 200, 255)

else:
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

    src1 = np.array([[(150, 500), (950, 500), (740, 280), (530, 280)]])

    mask = np.zeros_like(images[0])
    cv2.fillPoly(mask, src1, (255,255,255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = threshIt(mask, 200, 255)

print(len(images))

for i in range(len(images)):

    image = images[i]
    img = deepcopy(image)

    # Visualize points
    # cv2.circle(img, (280, 700), 5, (255, 0, 0), -1)
    # cv2.circle(img, (1100, 700), 5, (255, 0, 0), -1)
    # cv2.circle(img, (760, 480), 5, (255, 0, 0), -1)
    # cv2.circle(img, (600, 480), 5, (255, 0, 0), -1)

    # plt.imshow(img)
    # plt.show()
    # cv2.imshow("image", img)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # src = np.float32([[150, 500], [950, 500], [530, 280], [740, 280]])
    # dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
    # Hom = cv2.getPerspectiveTransform(src, dst)

    # HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    hbinary = threshIt(H, 10, 70)
    lbinary = threshIt(L, 120, 200)
    sbinary = threshIt(S, 120, 255)
    # cv2.imshow("H", 255 * hbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("L", 255 * sbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("S", 255 * lbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    yellow = np.zeros_like(img[:,:,0])
    yellow[(hbinary == 1) & (lbinary == 1) & (sbinary == 1)] = 1

    # cv2.imshow("yellow", 255 * yellow)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # RGB colour
    R = img[:,:,2]
    G = img[:,:,1]
    B = img[:,:,0]
    RGB = np.zeros_like(R)
    rbinary = threshIt(R, 200, 255)
    gbinary = threshIt(G, 200, 255)
    bbinary = threshIt(B, 200, 255)

    white = np.zeros_like(img[:,:,0])
    white[(rbinary == 1) & (gbinary == 1) & (bbinary == 1)] = 1
    # cv2.imshow("rbinary", 255*rbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("gbinary", 255*gbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("bbinary", 255*bbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("RGB", 255*RGB)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # YUV colour
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y = yuv[:,:,0]
    U = yuv[:,:,1]
    V = yuv[:,:,2]
    ybinary = threshIt(Y, 180, 255)
    ubinary = threshIt(U, 0, 120)
    vbinary = threshIt(V, 70, 200)
    # cv2.imshow("Y", Y)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("U", U)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("V", V)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # cv2.imshow("Y", 255*ybinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("U", 255*ubinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # cv2.imshow("V", 255*vbinary)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # combined1 = np.zeros_like(gray)
    # combined1[(sobel_x == 1)  | (sbinary == 1) | (rbinary == 1)] = 1

    # combined2 = np.zeros_like(gray)
    # combined2[((sobel_x == 1) & (sobel_y == 1)) | (rbinary == 1)] = 1

    # combined3 = np.zeros_like(gray)
    # combined3[(sbinary ==1) | (ubinary ==1)| (rbinary == 1 )] = 1

    # combined4 = np.zeros_like(gray)
    # combined4[(sobel_x == 1)  | (sbinary == 1) | (rbinary == 1 ) ] = 1

    combined5 = np.zeros_like(img[:,:,0])
    if select_data == 0:
        combined5[((white == 1) | ((yellow == 1) | (ubinary == 1))) & (mask == 1)] = 1
    else:
        combined5[((white == 1) | (yellow == 1)) & (mask == 1)] = 1

    # cv2.imshow("combined1", 255*combined5)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    # combined5 = cv2.warpPerspective(combined5, Hom, (200,400))

    # cv2.imshow("combined1-tran", 255*combined5)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()
    # combined6 = np.zeros_like(img[:, :, 0])
    # combined6[(mask == 0)] = 0
    combined5 = np.float32(combined5 * 255)
    sobelx = cv2.Sobel(combined5, cv2.CV_64F, 1, 0, ksize=5)
    # combined6 = np.float32(combined6 * 255)
    # sobelxleft = cv2.Sobel(combined6, cv2.CV_64F, 1, 0, ksize=5)
    # dst = cv2.Canny(np.uint8(sobelx), 50, 200, None, 3)
    # cv2.imshow('edges', dst)
    # cv2.waitKey(1)
    lines = cv2.HoughLines(np.uint8(sobelx), 1, np.pi / 100, 100)
    sobelx = np.float32(sobelx)
    cdst = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)
    xPositionsLeft = []
    # print('Left lines', len(lines))
    if lines is not None:
        for j in range(0, len(lines)):
            rho = lines[j][0][0]
            theta = lines[j][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x = int(-511*b/a + rho/a)
            # cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            xPositionsLeft.append(x)
    if lines is not None:
        # Find the left most line
        print('Left',xPositionsLeft)
        xleft = [xminval for xminval in xPositionsLeft if xminval < 700]
        # print(len(xleft))
        leftMost = int((len(xleft) / 2) + 0.5) - 1
        if leftMost >= 0:
            leftMost = xPositionsLeft.index(xleft[leftMost])
            rho = lines[leftMost][0][0]
            theta = lines[leftMost][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    lines = cv2.HoughLines(np.uint8(sobelx), 1, np.pi / 100, 150)
    xPositionsRight = []
    if lines is not None:
        for j in range(0, len(lines)):
            rho = lines[j][0][0]
            theta = lines[j][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x = int(-511*b/a + rho/a)
            xPositionsRight.append(x)
    #
    if lines is not None:
        # Find the left most line
        # leftMost = xPositionsLeft.index(min(xPositionsLeft))
        xright = [x for x in xPositionsRight if x > 700]
        rightMost = int((len(xright) / 2) + 0.5) - 1
        if rightMost >= 0:
            rightMost = xPositionsRight.index(xright[rightMost])
            # for j, _ in enumerate(lines[rightMost]):
            # if (leftMost - rightMost > 5):
            rho = lines[rightMost][0][0]
            theta = lines[rightMost][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(cdst, (700,512), (700,200), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('final', img)
    cv2.waitKey(1)

    # hom_inverse = np.linalg.inv(Hom)
    # new_points = []
    # if len(final_points) == 2:
    # 	for num in range(2):
    # 		new_points.append(np.matmul(hom_inverse, [final_points[num][0][0], final_points[num][0][1], 1]))
    # 		new_points.append(np.matmul(hom_inverse, [final_points[num][1][0], final_points[num][1][1], 1]))
    #
    # 	cv2.circle(img, (int(new_points[0][0] / new_points[0][2]), int(new_points[0][1] / new_points[0][2])), 5,
    # 			   (255, 0, 0), -1)
    # 	cv2.circle(img, (int(new_points[1][0] / new_points[1][2]), int(new_points[1][1] / new_points[1][2])), 5,
    # 			   (255, 0, 0), -1)
    # 	cv2.circle(img, (int(new_points[2][0] / new_points[2][2]), int(new_points[2][1] / new_points[2][2])), 5,
    # 			   (255, 0, 0), -1)
    # 	cv2.circle(img, (int(new_points[3][0] / new_points[3][2]), int(new_points[3][1] / new_points[3][2])), 5,
    # 			   (255, 0, 0), -1)
    #
    # 	cv2.imshow("poly_points", img)
    # 	cv2.waitKey(1)