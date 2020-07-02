# Vehicle Detection in a video
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is a software pipeline used to detect cars in a video. 

## Vehicle Detection Project

The steps to detect vehicles in a video are the following:

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
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

Note: I wrote my code originally in [a jupyter notebook file](https://github.com/kkufieta/CarND-Vehicle-Detection/blob/master/vehicle_detection.ipynb). All code lines I refer to from now on are found in the extracted [python file](https://github.com/kkufieta/CarND-Vehicle-Detection/blob/master/vehicle_detection.py).

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images (Lines 50-80).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1a]

Next, I implemented functions to extract HOG features (`get_hog_features`), binned color features (`bin_spatial`) and color histograms (`color_hist`) (Lines 98-142). I plotted a few examples simply using the original image (or grey image in the case of HOG features, Lines 144-237):

![alt text][image1b]
![alt text][image1c]

After that, I defined the `extract_features` function that extracts all the features for an image (Lines 247-313). I extracted HOG features from the training images on Lines 359-383.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. First I performed a grid search on the parameters: 
* All color spaces
* pixels per cells = 4, 8 and 16
* blocks per cell = 1 and 2
* orientations = 9 and 12. 

I trained the `LinearSVC` classifier with 500 data samples and tested them with 100 data samples, based on all those combinations, and received tables with test accuracies. Let's show them off, since I've done the work (not that it proved useful). I deleted the code for the grid search since it didn't prove useful, but it is available in older commits. 

Note: The values within the table divided by a backslash are for 1 and 2 cells/block, respectively. The last table shows the number of features for each parameter combination.

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

The validation accuracies are very similar. The factor that varies most is how big the feature vector turns out. I decided that HSV with 9 orientations and 4 pixels/cell has one of the highest accuracies while maintaining a manageable feature vector size of 24300, so I decided to explore that combination further.

Next, I explored different color spaces visually and how their HOG features look like. I explored particularly HSV to find out if the channels give redundant HOG information, or if all three prove to be useful features. Note: I deleted this code as well because it didn't prove useful, but it is available in my commit history.

Here are a few examples using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(4, 4)` and `pixels_per_cell=(8, 8)`, and `cells_per_block=(2, 2)`:

`pixels_per_cell=(4,4)`:
![alt text][image2a]
`pixels_per_cell=(8,8)`:
![alt text][image2b]

You can see how much more detailed the HOG features look when using `pixels_per_cell=(4,4)` instead of `pixels_per_cell=(8,8)`.

Based on the images I plotted above and the values I received from my grid search, I decided to work with HSV and both 4 and 8 pixels/cell. I worked for 50 hours (not a joke) with various combinations, one of which was this one:

* H-channel: 4 pixels/cells, 1 cell/block, 9 orientations
* S-channel: 4 pixels/cells, 1 cell/block, 9 orientations
* S-channel: 8 pixels/cells, 1 cell/block, 9 orientations
* V-channel: 8 pixels/cells, 1 cell/block, 9 orientations

This was performing quite alright, but not good enough to pass the project submission. I was about to quit, when my friend Kedar told me about his particularly well performing parameter combination, which I decided to give a try:

* All YUV channels
* orient: 7
* pixels/cell: 8
* cells/block: 2

It wasn't perfect. Not knowing what else to try, I finally added binned color features and color histograms. That did the trick! The cars were finally almost perfectly classified, but so were the shadows on the streets. After a little bit of research, I found in the [hog documentation](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog) that I could switch on `transform_sqrt` to avoid detecting shadows. I quickly realized that wouldn't work with YUV, since I get negative values in the Y channel, so I switched to YCrCb which resulted in this parameter set:

* All YCrCb channels, orient: 7, pixels/cell: 8, cells/block: 2
* binned color features and color histograms, `spatial_size=(32, 32)`, `hist_bins=32`
* `transform_sqrt=True`

I actually tried HSV before YCrCb, but that made the performance really bad. Then I realized that YUV and YCrCb look quite similar, and the latter finally did the trick! The power law compression (`transform_sqrt`) worked very well to avoid detecting shadows and railings.

Lessons learned: My grid search and initial choice of parameters was useless.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First, I split my data into training and validation data (Lines 323-346). I did that before shuffling them, so I could have data samples that are different for training and testing. If I would have shuffled before splitting, I would have had very similar images in both training and testing set, and validation accuracy would show a 98-99% accuracy, which is not correct.

Next, I applied the feature extraction on all the data (Lines 352-384). After that, I scaled the features (Lines 388-404). Finally I shuffled them (Lines 405-412).

I trained my classifier with a penalty parameter of `C=3e-6` to avoid overfitting (Lines 415-430).

Once I trained it, I tested its performance on both the validation set and part of the training set (Lines 432-443). Those have to be compared to detect overfitting. I used both test and training accuracy to optimize the penalty parameter `C` until I received a result I was happy with.

Once trained, I pickled my classifier (Lines 446-463).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the sliding window approach with HOG-subsampling from Udacity (Lines 503-595). It is much faster than a simple sliding window approach, reduces video processing from 10 hours to 1 hour (for the 50 second project video). 

The sliding window approach reads in the area of the image that needs to be processed, and runs the HOG algorithm on it (Lines 541-546), saving HOG features for the entire image. Afterwards, it samples the image by moving the sliding window over it (Lines 549-593), extracting the respective indices to get the corresponding HOG values (Lines 554-560). Additionaly, it runs the binned color features and the color histogram on each window (Lines 570-572), combines all features and uses the classifier to predict whether there is a car within the current window, or not (Line 581). Once classified, it moves the window by a number of pixels defined in `cells_per_step`, and repeats the process.

This sliding window approach does allow to look for cars in windows with various sizes. This can be specified by the `scale` parameter. It scales the image to be larger or smaller, while the sliding window size remains the same size. When the image is scaled to appear bigger, we're effectively scanning a smaller portion of the image at a time (smaller windows). When the image is scaled to be smaller, we're effectively scanning a larger portion of the image at a time (larger windows). This is important, because cars appear larger or smaller in the image, based on their location.

I found through extensive testing that overlapping the windows by at least 75% gives good result. Also window sizes of 150, 130, 100, 75 and 50 pixels give quite good results and cover cars close by as well as cars further away. This led me to choose the following window sizes (Lines 617-619):

* scale=2.3, cells_per_step=1
* scale=1.9, cells_per_step=1
* scale=1.4, cells_per_step=2
* scale=1, cells_per_step=2

I chose to search the image between roughly `400px` to `650px` in y-direction, and in the entire x-direction span (Lines 615-616). That way I could focus on the street only, and avoid getting false positives in the sky and trees. No need to scan the hood of the car, either.

This image shows what my sliding window algorithm covers in an image:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

First I trained lots of classifiers with really bad and not so bad (but still not good enough) performance. One example would be my custom HSV parameter set that I've described above, which leads to these results:
![alt text][image_bad1]

I was trying hard to get good results for 50 hours, until I found my magic parameter combination as explained above (tl;dr: Parameters hinted from a friend, subsequent small changes led to the desired results). Once I had the magic parameter combination, I added spatially binned features and color histograms. I enabled power law compression in my HOG features extraction to avoid detecting shadows. I trained my classifier with a penalty parameter of `C=3e-6` to avoid overfitting (tuned by comparing test and training accuracy). 

I knew my parameter combination was working once I tested my pipeline and saw those beautiful pictures that resembled the performance from the lecture quizzes:

![alt text][image4a]
![alt text][image4b]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1] or you can view it directly on [YouTube][video_link].


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used Udacities idea to calculate a heatmap (all pixels that are within 'detected' boxes are summarized and are considered 'heat') and create bounding boxes using `label` from the `scipy.ndimage.measurements` library (Lines 656-706). 

As I'm scanning all images for cars, I make a list of all the boxes that have a car detected. From those boxes, I create a heatmap and threshold the heatmap with a certain value (here: 2). All this is implemented in the function `pipeline` (Lines 722-791).

Next, while processing the video, I keep a rolling list and a sum of the past 12 heatmaps. I updated both the list and the sum with every new frame (add the new values, delete the oldest). This is implemented in the function `car_finding` (Lines 817-816). The sum of heatmaps is thresholded by the number of saved frames (here: 12). The bounding boxes are calculated based on the averaged sum of heatmaps from the last 12 frames. 

This method filters out false positives, while keeping the car detections, if correctly tuned.

### Here are the six test images and their corresponding heatmaps:

For every example, you can see all the detected windows (the blue boxes in the top image), the final heatmap for the frame (lower right), and the resulting bounding boxes for the current frame (lower left).

![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]
![alt text][image5e]
![alt text][image5f]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I failed to find the correct parameters with my grid search and way of thinking. It is still difficult for me to understand the "art" of training a neural network and I received only good results when following closely to what was provided from Udacity, instead of trying to freestyle it on my own. More practice is needed.

Next, I might get better results if I would tweak my parameters more, and would do more test runs. But after close to 60 hours of working on this project, I'm postponing that to the future. I believe a different choice of heatmap thresholds or window sizes might lead to even better results.

Testing my algorithm on the project video revealed two flaws:
* Cars appearing on the side are detected rather late. I might be able to fix that by augmenting my dataset and e.g. shifting the image so it learns to classify cars not only when they're fully in the picture, but also when only half the car is in the image.
* Even though my algorithm barely reacts to shadows, there was a moment when it did so and extended the bounding box for the black car too much. In real life that would likely lead to emergency breaking, so fixing my algorithm to perform better when faced with shadows is crucial.

Besides having to test the pipeline in more scenarios with shadows, I can imagine it to fail in different weather conditions, in more complex scenarios (like in a city).


