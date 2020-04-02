# ENPM-673-Lane-Detection
The project aimed to perform Lane Detection and Tyrn Prediction to mimic Lane Departure Warning Systems used in
Self Driving Cars. We perform color segmentation and calculate histogram of lane pixels to detect lanes and fit a polygon over it. 

## Run the code

Run the following command to enhance the video in Part 1. 

``` python Problem1.py --convert=True ```

Run the following command to run the lane detection code:

``` python lane_detector.py <data_selection> ```

Set <data_selection> = 1 for selecting the first data set and <data_selection> = 2 for the second data set. 

Note: The code needs the data-sets to be placed in the current directory.

## Results:

A sample of output videos on given dataset:

![Result1](Results/demo/gif1.gif)
![Result2](Results/demo/gif2.gif)
