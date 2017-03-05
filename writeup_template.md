##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1a]: ./examples/car_not_car.png
[image1b]: ./examples/example_hist.png
[image1c]: ./examples/example_hog_spatial.png
[image2a]: ./examples/03_orient_9_pixpcell_4_cellpblock_2.png
[image2b]: ./examples/03_orient_9_pixpcell_8_cellpblock_2.png
[image3]: ./examples/sliding_windows.png
[image4a]: ./examples/sliding_window.png
[image4b]: ./examples/all_scales_combined.png
[image_bad1]: ./examples/bad_performance.png
[image5a]: ./examples/bboxes_and_heat_01.png
[image5b]: ./examples/bboxes_and_heat_02.png
[image5c]: ./examples/bboxes_and_heat_03.png
[image5d]: ./examples/bboxes_and_heat_04.png
[image5e]: ./examples/bboxes_and_heat_05.png
[image5f]: ./examples/bboxes_and_heat_06.png
[image6]: ./examples/all_scales_combined.png
[image7]: ./examples/11_with_heat.png
[video1]: ./processed_video.mp4
[video_link]: https://www.youtube.com/watch?v=Nj_qJD7Rbec

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

Note: I wrote my code originally in [a jupyter notebook file](https://github.com/kkufieta/CarND-Vehicle-Detection/blob/master/vehicle_detection.ipynb). All code lines I refer to from now on are found in the extracted [python file](https://github.com/kkufieta/CarND-Vehicle-Detection/blob/master/vehicle_detection.py).

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images (Lines 50-80).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1a]

Next, I implemented functions to extract HOG features, binned color features and color histograms (Lines 98-142). I plotted a few examples simply using the original image (or grey image in the case of HOG features):

![alt text][image1b]
![alt text][image1b]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is are a few examples using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(4, 4)` and `pixels_per_cell=(8, 8)`, and `cells_per_block=(2, 2)`:

![alt text][image2a]
![alt text][image2b]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. 

First, I performed a grid search on the parameters: All color spaces, pixels per cells = 4, 8 and 16, blocks per cell = 1 and 2, orientations = 9 and 12. I trained classifiers with 500 data samples based on all those combinations, and received tables with test accuracies. Let's show them off, since I've done the work (not that it proved useful, which is why I deleted it from my code as well. The code is available in older commits.) 

| RGB | 9           | 12          |
| --- |:-----------:| -----------:|
| 4   | 0.97 / 0.97 | 0.95 / 0.95 |
| 8   | 0.96 / 0.95 | 0.95 / 0.95 |
| 16  | 0.95 / 0.91 | 0.94 / 0.89 |

| HSV | 9           | 12          |
| --- |:-----------:| -----------:|
| 4   | 0.97 / 0.94 | 0.94 / 0.92 |
| 8   | 0.95 / 0.91 | 0.95 / 0.92 |
| 16  | 0.94 / 0.87 | 0.93 / 0.89 |

| LUV | 9           | 12          |
| --- |:-----------:| -----------:|
| 4   | 0.94 / 0.94 | 0.93 / 0.93 |
| 8   | 0.95 / 0.96 | 0.94 / 0.94 |
| 16  | 0.94 / 0.91 | 0.94 / 0.90 |

| YUV | 9           | 12          |
| --- |:-----------:| -----------:|
| 4   | 0.96 / 0.96 | 0.94 / 0.93 |
| 8   | 0.95 / 0.96 | 0.95 / 0.92 |
| 16  | 0.96 / 0.91 | 0.93 / 0.91 |

| YCrCb | 9           | 12          |
| ----- |:-----------:| -----------:|
| 4     | 0.95 / 0.96 | 0.93 / 0.92 |
| 8     | 0.94 / 0.95 | 0.91 / 0.93 |
| 16    | 0.96 / 0.91 | 0.93 / 0.91 |

| number features | 9             | 12            |
| --------------- |:-------------:| -------------:|
| 4               | 24300 / 73008 | 32400 / 97344 |
| 8               | 5292 / 10800  | 7056 / 14400  |
| 16              | 972 / 432     | 1296 / 576    |

As I've showed above, I examined color spaces and other parameteres manually and looked at the HOG plot. I then decided to choose a parameter and color space combination that would help me, a human, to distinguish between cars and non-cars. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4a]
![alt text][image4b]

![alt text][image_bad1]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]
or [youtube][video_link]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]
![alt text][image5e]
![alt text][image5f]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

