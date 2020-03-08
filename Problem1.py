import cv2
import argparse


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
    out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame_width, frame_height))
    alpha = 2
    beta = 50
    # image = cv2.imread('frame0.jpg')
    for i in range(619):
    #     print('Running for image: frame' + str(i) + '.jpg')
        image = cv2.imread('frame' + str(i) + '.jpg')
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        out.write(new_image)
        cv2.imshow('New Image', new_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if need to convert video to images.')
    parser.add_argument('--convert', default=False,
                      help='Convert video to images first?')
    args = parser.parse_args()
    if args.convert:
        convert_video_to_images()

    increase_contrast_and_brightness()