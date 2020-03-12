import numpy as np
import cv2
from copy import deepcopy
import numpy.linalg as la 
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
    out = cv2.VideoWriter('HistogramOutputData2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (images[0].shape[1], images[0].shape[0]))

    src = np.float32([[280, 700], [1100, 700], [600, 480], [760, 480]])
    dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])

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
    out = cv2.VideoWriter('HistogramOutputData1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                          (images[0].shape[1], images[0].shape[0]))

    src = np.float32([[150, 500], [950, 500], [530, 280], [740, 280]])
    dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])

#print(len(images))

for i in range(len(images)):
    #print(i)
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

    # cv2.imshow("yellow", 255 * yellow) np.float32([[280, 700], [1100, 700], [600, 480], [760, 480]])
    # dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
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
    # combined5 = np.float32(combined5 * 255)

    src = np.float32([[150, 500], [950, 500], [530, 280], [740, 280]])
    # src = np.float32([[280, 700], [1100, 700], [600, 480], [760, 480]])
    dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
    # src = np.float32([[150, 500], [950, 500], [530, 280], [740, 280]])
    # src = np.float32([[280, 700], [1100, 700], [600, 480], [760, 480]])
    # dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
    Hom = cv2.getPerspectiveTransform(src, dst)

    binary_warped = cv2.warpPerspective(combined5, Hom, (200,400))

    # binary_warped = combined5
    
    # binary_warped = cv2.Sobel(combined5, cv2.CV_64F, 1, 0, ksize=5)
    # cv2.waitKey(1)
    #print(binary_warped.shape)

    # cv2.waitKey(0)

    histogram = np.sum(binary_warped, axis=0)
    #print(histogram.shape)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #print("x_base:")
    #print(leftx_base, rightx_base)

    nwindows = 20
    minpix=50

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    margin = 35
    # Step through the windows one by one
    for window in range(nwindows):
        # print("window", window+1)
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        # print("nonzero")
        # print(len(nonzero[0]))
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # print(good_left_inds.shape)
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # print(rightx.shape)
    # print(rightx.shape)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    for k in range(len(ploty)):
        cv2.circle(out_img, (int(left_fitx[k]), int(ploty[k])), 5, (255, 0, 0), -1)
        cv2.circle(out_img, (int(right_fitx[k]), int(ploty[k])), 5, (255, 0, 0), -1)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    Hinv = cv2.getPerspectiveTransform(dst, src)
    # Hinv = la.inv(Hom)

    newwarp = cv2.warpPerspective(out_img, Hinv, (img.shape[1], img.shape[0]))

    # newwarp = cv2.warpPerspective(color_warp, Hinv, (img.shape[1], img.shape[0])) 
    img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    
    # roc_left =  ((1 + (2*left_fit_m[0]*yRange*y_metres + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    # roc_right =  ((1 + (2*right_fit_m[0]*yRange*y_metres + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])

    # leftCurvature = roc_left / 1000
    # rightCurvature = roc_right/ 1000

    xMax = img.shape[1]
    yMax = 0
    vehicleCenter = xMax / 2
    # print("left fit", left_fit)
    # print("right fit", right_fit)

    lineLeft = left_fit[0]*yMax**2 + left_fit[1]*yMax + left_fit[2]

    lineRight = right_fit[0]*yMax**2 + right_fit[1]*yMax + right_fit[2]
    lineMiddle = lineLeft + (lineRight - lineLeft)/2

    pt1 = np.matmul(Hinv, [lineMiddle,yMax,1])
    pt2= np.matmul(Hinv, [0, binary_warped.shape[0],1])
    print(" line middle", int(pt1[0]/pt1[2]))
    print("vehicle centre", int(vehicleCenter))
    diffFromVehicle = int(pt1[0]/pt1[2]) - int(vehicleCenter)



    cv2.circle(img, (int(pt1[0]/pt1[2]), int(pt1[1]/pt1[2])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(vehicleCenter), int(pt2[1]/pt2[2])), 5, (255, 0, 0), -1)

    if diffFromVehicle >=30:
        print("Right")
        cv2.putText(img, 'Right turn', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    elif diffFromVehicle <= 21:
        print("Left")
        cv2.putText(img, 'Left turn', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    else:
        print("Straight")
        cv2.putText(img, 'Straight', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)





    cv2.imshow("out_img", img)
    cv2.waitKey(1)
    out.write(img)
out.release()

# plt.show()

    
    
    # Fit a second order polynomial to each
    # left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    # right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)