import numpy as np
import cv2
from copy import deepcopy
import os
import argparse


def binary_thresh(img, thresh_min, thresh_max):
    xbinary = np.zeros_like(img)
    xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return xbinary

def lane_detection(select_data):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_path, dir_name = os.path.split(current_dir)
    image_dir = os.path.join(current_dir, "data_1/data")
    
    images = []
    image_names = []
    
    select_data = select_data - 1
    
    if select_data == 1:
        K = np.asarray([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                        [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        D = np.asarray([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
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
        mask = binary_thresh(mask, 200, 255)
        out = cv2.VideoWriter('LaneDetectionOutputData2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                              (images[0].shape[1], images[0].shape[0]))
        #Define Region of interest
        src = np.float32([[280, 700], [1100, 700], [600, 480], [760, 480]])
        dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
    
    else:
    
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
    
        src1 = np.array([[(150, 500), (950, 500), (740, 280), (530, 280)]])
    
        mask = np.zeros_like(images[0])
        cv2.fillPoly(mask, src1, (255,255,255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = binary_thresh(mask, 200, 255)
        out = cv2.VideoWriter('LaneDetectionOutputData1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                              (images[0].shape[1], images[0].shape[0]))
        #Region of interest
        src = np.float32([[150, 500], [950, 500], [530, 280], [740, 280]])
        dst = np.float32([[0, 400], [200, 400], [0, 0] , [200, 0]])
    
    #print(len(images))
    
    for i in range(len(images)):
        #print(i)
        image = images[i]
        img = deepcopy(image)
        #apply HLS thresholding for yellow line
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
    
        hbinary = binary_thresh(H, 10, 70)
        lbinary = binary_thresh(L, 120, 200)
        sbinary = binary_thresh(S, 120, 255)
    
        yellow = np.zeros_like(img[:,:,0])
        yellow[(hbinary == 1) & (lbinary == 1) & (sbinary == 1)] = 1
        # apply RGB thresholding for white line
        R = img[:,:,2]
        G = img[:,:,1]
        B = img[:,:,0]
        rbinary = binary_thresh(R, 200, 255)
        gbinary = binary_thresh(G, 200, 255)
        bbinary = binary_thresh(B, 200, 255)
    
        white = np.zeros_like(img[:,:,0])
        white[(rbinary == 1) & (gbinary == 1) & (bbinary == 1)] = 1
    
        # apply YUV thresholding for illumination
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        U = yuv[:,:,1]
        ubinary = binary_thresh(U, 0, 120)
    
        combined5 = np.zeros_like(img[:,:,0])
        if select_data == 1:
            combined5[((white == 1) | ((yellow == 1) | (ubinary == 1))) & (mask == 1)] = 1
        else:
            combined5[((white == 1) | (yellow == 1)) & (mask == 1)] = 1
        Hom = cv2.getPerspectiveTransform(src, dst)
    
        binary_warped = cv2.warpPerspective(combined5, Hom, (200,400))
    
        histogram = np.sum(binary_warped, axis=0)
        #print(histogram.shape)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        nwindows = 20
        minpix = 50
    
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
        # Checking for lane candidates in window
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
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
    
        # Generate curve
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create polygon
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
    
        # Overlay polygon mesh
        img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
        yMin = 0
        lineLeft = left_fit[0]*yMin**2 + left_fit[1]*yMin + left_fit[2]
        lineRight = right_fit[0]*yMin**2 + right_fit[1]*yMin + right_fit[2]
        
        # Center of top of polygon
        lineMiddle = lineLeft + (lineRight - lineLeft)/2
    
        yMax = binary_warped.shape[1] - 1
        lineLeft = left_fit[0] * yMax ** 2 + left_fit[1] * yMax + left_fit[2]
        lineRight = right_fit[0] * yMax ** 2 + right_fit[1] * yMax + right_fit[2]
        
        # Center of bottom of polygon
        vehicleCenter = lineLeft + (lineRight - lineLeft) / 2
        
        # Deviation from center
        diffFromVehicle = int(lineMiddle) - int(vehicleCenter)
    
        if diffFromVehicle > 10:
            print("Right")
            cv2.putText(img, 'Right turn', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        elif diffFromVehicle < -10:
            print("Left")
            cv2.putText(img, 'Left turn', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        else:
            print("Straight")
            cv2.putText(img, 'Straight', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
        cv2.imshow("out_img", img)
        cv2.waitKey(1)
        out.write(img)
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine which data set to run for')
    parser.add_argument('dataset', choices=['1', '2'],
                        help='Select which data set to run for, 1 or 2?')
    args = parser.parse_args()
    lane_detection(int(args.dataset))