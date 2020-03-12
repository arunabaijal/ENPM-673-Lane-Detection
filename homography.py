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

select_data = 0

K = np.asarray([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                [0.000000e+00, 9.019653e+02, 2.242509e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00]])

D = np.asarray([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

width = 0
height = 0
midval = 0
threshleft = 0
threshright = 0

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
    out = cv2.VideoWriter('PolyOverlayData2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (images[0].shape[1], images[0].shape[0]))
    width = images[0].shape[1]
    height = images[0].shape[0]
    midval = 500
    threshleft = 150
    threshright = 70

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
    out = cv2.VideoWriter('PolyOverlayData1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (images[0].shape[1], images[0].shape[0]))
    width = images[0].shape[1]
    height = images[0].shape[0]
    midval = 270
    threshleft = 120
    threshright = 120

print(len(images))

for i in range(len(images)):

    image = images[i]
    img = deepcopy(image)

    # Visualize points
    # cv2.circle(img, (280, 700), 5, (255, 0, 0), -1)
    # cv2.circle(img, (1100, 700), 5, (255, 0, 0), -1)
    # cv2.circle(img, (760, 480), 5, (255, 0, 0), -1)
    # cv2.circle(img, (600, 480), 5, (255, 0, 0), -1)

    # HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]

    hbinary = threshIt(H, 10, 70)
    lbinary = threshIt(L, 120, 200)
    sbinary = threshIt(S, 120, 255)

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

    # YUV colour
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y = yuv[:,:,0]
    U = yuv[:,:,1]
    V = yuv[:,:,2]
    ybinary = threshIt(Y, 180, 255)
    ubinary = threshIt(U, 0, 120)
    vbinary = threshIt(V, 70, 200)

    combined5 = np.zeros_like(img[:,:,0])
    if select_data == 0:
        combined5[((white == 1) | ((yellow == 1) | (ubinary == 1))) & (mask == 1)] = 1
    else:
        combined5[((white == 1) | (yellow == 1)) & (mask == 1)] = 1
    combined5 = np.float32(combined5 * 255)
    sobelx = cv2.Sobel(combined5, cv2.CV_64F, 1, 0, ksize=5)
    # cv2.waitKey(1)
    lines = cv2.HoughLines(np.uint8(sobelx), 1, np.pi / 100, threshleft)
    sobelx = np.float32(sobelx)
    cdst = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)
    xPositionsLeft = []
    params = []
    # print('Left lines', len(lines))
    points = []
    overlay = img.copy()
    if lines is not None:
        for j in range(0, len(lines)):
            rho = lines[j][0][0]
            theta = lines[j][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x1 = int(-(height - 1) * b / a + rho / a)
            x2 = int(-midval * b / a + rho / a)
            # cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            if 0 <= x1 < width:
                xPositionsLeft.append([x1,x2])
                params.append([rho, theta])
        # Find the left most line
        xleft = [[xminval, xmaxval] for xminval, xmaxval in xPositionsLeft if xminval < 650 and xmaxval < 650]
        # xright = [[xminval, xmaxval] for xminval, xmaxval in xPositionsLeft if xminval > 700 and xmaxval > 700]
        # print(len(xleft))
        indexes = [xleft]
        for item in indexes:
            leftMost = int((len(item) / 2) + 0.5) - 1
            if leftMost >= 0:
                leftMost = xPositionsLeft.index(item[leftMost])
                rho = params[leftMost][0]
                theta = params[leftMost][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x1 = int(-(height-1) * b / a + rho / a)
                x2 = int(-midval * b / a + rho / a)
                pt1 = (x1, height-1)
                pt2 = (x2, midval)
                cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                points.append(pt1)
                points.append(pt2)
    lines = cv2.HoughLines(np.uint8(sobelx), 1, np.pi / 100, threshright)
    if lines is not None:
        for j in range(0, len(lines)):
            rho = lines[j][0][0]
            theta = lines[j][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x1 = int(-(height - 1) * b / a + rho / a)
            x2 = int(-midval * b / a + rho / a)
            # cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            if 0 <= x1 < width:
                xPositionsLeft.append([x1,x2])
                params.append([rho, theta])
        # Find the left most line
        # xleft = [[xminval, xmaxval] for xminval, xmaxval in xPositionsLeft if xminval < 700 and xmaxval < 700]
        xright = [[xminval, xmaxval] for xminval, xmaxval in xPositionsLeft if xminval > 650 and xmaxval > 650]
        # print(len(xleft))
        indexes = [xright]
        for item in indexes:
            leftMost = int((len(item) / 2) + 0.5) - 1
            if leftMost >= 0:
                leftMost = xPositionsLeft.index(item[leftMost])
                rho = params[leftMost][0]
                theta = params[leftMost][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x1 = int(-(height-1) * b / a + rho / a)
                x2 = int(-midval * b / a + rho / a)
                pt1 = (x1, height-1)
                pt2 = (x2, midval)
                cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                points.append(pt1)
                points.append(pt2)
    if len(points) == 4:
        cv2.line(img, points[0], points[2], (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(img, points[1], points[3], (0, 0, 255), 3, cv2.LINE_AA)
        temp = points[2]
        points[2] = points[3]
        points[3] = temp
        cv2.fillPoly(overlay, pts=[np.array(points)], color=(0, 0, 255))

        alpha = 0.4  # Transparency factor.

        # Following line overlays transparent rectangle over the image
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        centre_x1 =  (points[0][0] + points[3][0])/2
        print("centre x1", centre_x1)
        centre_x2 = (points[1][0] + points[2][0])/2
        print("centre x2", centre_x2)
        centre_pt1 = (centre_x1 , height-1)
        centre_pt2 = (centre_x2, midval)
        cv2.line(img, centre_pt1, centre_pt2, (0, 255, 0), 3, cv2.LINE_AA)

        if float(centre_pt2[0] - centre_pt1[0]) ==0:
        	print("0000000000000000000000000000 Straight")

        else:
            slope_centre = float(centre_pt2[1] - centre_pt1[1]) /float(centre_pt2[0] - centre_pt1[0])
            print(slope_centre)
            if slope_centre <0:
            	angle = 180 - math.degrees(math.atan(abs(slope_centre)))
            else:
            	angle = math.degrees(math.atan(slope_centre))
            print("angle", angle)
            if 82<angle <99 :
                print("xxxxxxxxxxxxxxxxxxxxxxxxxx Straight")
            elif angle < 82:
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxx Left")

            elif angle > 99:
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx right")
            else:
                print("yyyyyyyyyyyyyyyyyyyyyyyyyyyy Straight")




    cv2.imshow('final', img)
    cv2.waitKey(1)
    out.write(img)
out.release()

