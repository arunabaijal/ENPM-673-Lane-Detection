import cv2
import argparse
import numpy as np


def convert_video_to_images():
    print('Converting video to images..')
    vidcap = cv2.VideoCapture('Night Drive - 2689.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      count += 1


def increase_contrast_and_brightness():
    img = cv2.imread('frame0.jpg')
    frame_width = img.shape[1]
    frame_height = img.shape[0]
    out = cv2.VideoWriter('resultHistogramEqualization.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))
    out_contrast = cv2.VideoWriter('resultContrastBrightness.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))
    alpha = 2
    beta = 50
    # image = cv2.imread('frame0.jpg')
    for i in range(619):
        image = cv2.imread('frame' + str(i) + '.jpg')
        ret, image = cv2.threshold(image, 10, 255, cv2.THRESH_TOZERO)
        image_r = image[:,:,0]
        image_g = image[:,:,1]
        image_b = image[:,:,2]
        new_image_r = cv2.equalizeHist(image_r)
        new_image_g = cv2.equalizeHist(image_g)
        new_image_b = cv2.equalizeHist(image_b)
        new_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        new_image[:,:,0] = new_image_r
        new_image[:,:,1] = new_image_g
        new_image[:,:,2] = new_image_b
        out.write(new_image)
        cv2.imshow('histogram', new_image)
        cv2.waitKey(1)
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        out_contrast.write(new_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if need to convert video to images.')
    parser.add_argument('--convert', default=False,
                      help='Convert video to images first?')
    args = parser.parse_args()
    if args.convert:
        convert_video_to_images()

    increase_contrast_and_brightness()