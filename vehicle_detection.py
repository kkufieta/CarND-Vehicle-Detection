
# coding: utf-8

# # Vehicle Detection & Tracking
# 
# The task in this project is to detect and track vehicles in images and videos.
# 
# To do this, we have to decide which features we'd like to extract out of the input images. Next, we have to train a classifier. Eventually, we have to test the algorithm on input images and video streams.
# 
# The goals / steps of this project are the following:
# 
# * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.
# 
# ## Import libraries

# In[2]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip


# ## Load Data
# 
# Images for training the classifier are divided up into vehicles & non-vehicles, and can be found in the folders `dataset_vehicles` and `dataset_non_vehicles`.
# 
# Images used for testing images can be found in the folder `test_images`.
# 
# Let's load the data.

# In[4]:

# Import all image paths for the images used to train the classifier
car_paths = glob.glob('dataset_vehicles/*/*.png')
not_car_paths = glob.glob('dataset_non_vehicles/*/*.png')
print('Number of car images: ', len(car_paths))
print('Number of not-car images: ', len(not_car_paths))
example_image = mpimg.imread(car_paths[0])
print('Image shape: ', example_image.shape)
print('Image data type: ', type(example_image), ", ", example_image.dtype)

# Import all image paths for the images to test your classifier on
test_paths = glob.glob('test_images/*.jpg')

# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(car_paths))
notcar_ind = np.random.randint(0, len(not_car_paths))
    
# Read in car / not-car images
car_image = mpimg.imread(car_paths[car_ind])
notcar_image = mpimg.imread(not_car_paths[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.show()


# ## Feature Extraction
# 
# Define functions to extract the following features:
# * Histogram of Oriented Gradients (HOG)
# * Binned color features
# * Color histogram features

# In[9]:

# Function returns HOG features and visualization
# orient: Number of orientation bins
# pix_per_cell: (2-tuple, (int, int)). Size of a cell (in pixels).
# cell_per_block: (2-tuple, (int, int)). Number of cells in each block.
# vis: bool. Return an image of the HOG?
# feature_vec: bool. Return the data as a feature vector by calling .ravel() on it?

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
        
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec, 
                                 transform_sqrt=True)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient,
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block),
                        visualise=vis, feature_vector=feature_vec,
                      transform_sqrt=True)
        return features
    
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256), visualize=False):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    if visualize:        
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2    
        # Return the individual histograms, bin_centers and feature vector
        return rhist, ghist, bhist, bin_centers, hist_features
    else:
        return hist_features


# Plot a few examples to show how this looks like.

# In[28]:

# Generate a random index to look at a car image
car_ind = np.random.randint(0, len(car_paths))
not_car_ind = np.random.randint(0, len(not_car_paths))
# Read in the image. Let's work with values between 0 to 255
car_image = mpimg.imread(car_paths[car_ind])
not_car_image = mpimg.imread(not_car_paths[not_car_ind])

car_gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
not_car_gray = cv2.cvtColor(not_car_image, cv2.COLOR_RGB2GRAY)
print(car_gray.shape)

# Define HOG parameters
orient = 9
pix_per_cell = 4
cell_per_block = 2

# Get the HOG features and visualization
car_hog_features, car_hog_image = get_hog_features(car_gray, orient, pix_per_cell, 
                                                   cell_per_block, vis=True, 
                                                   feature_vec=False)
not_car_hog_features, not_car_hog_image = get_hog_features(not_car_gray, orient, 
                                                           pix_per_cell, cell_per_block, 
                                                           vis=True, feature_vec=False)

# Compute color histogram features
car_image = np.uint8(car_image*255)
not_car_image = np.uint8(not_car_image*255)
car_rh, car_gh, car_bh, car_bincen, car_feature_vec = color_hist(car_image, visualize=True)
not_car_rh, not_car_gh, not_car_bh, not_car_bincen, not_car_feature_vec = color_hist(not_car_image, 
                                                                                     visualize=True)

# Compute spatially binned features
car_bin_spatial = bin_spatial(car_image)
not_car_bin_spatial = bin_spatial(not_car_image)


# Plot the examples
fig = plt.figure(figsize=(20,20))
plt.subplot(321)
plt.imshow(car_image)
plt.title('Example Car Image', fontsize=20)
plt.subplot(322)
plt.imshow(not_car_image)
plt.title('Example Not-Car Image', fontsize=20)
plt.subplot(323)
plt.imshow(car_hog_image, cmap='gray')
plt.title('HOG Car Visualization', fontsize=20)
plt.subplot(324)
plt.imshow(not_car_hog_image, cmap='gray')
plt.title('HOG Not-Car Visualization', fontsize=20)
plt.subplot(325)
plt.plot(car_bin_spatial)
plt.title('Spatially Binned Features Car', fontsize=20)
plt.subplot(326)
plt.plot(not_car_bin_spatial)
plt.title('Spatially Binned Features Not-Car', fontsize=20)
plt.show()


fig = plt.figure(figsize=(10,5))
plt.subplot(131)
plt.bar(car_bincen, car_rh[0])
plt.xlim(0, 256)
plt.title('R Histogram Car')
plt.subplot(132)
plt.bar(car_bincen, car_gh[0])
plt.xlim(0, 256)
plt.title('G Histogram Car')
plt.subplot(133)
plt.bar(car_bincen, car_bh[0])
plt.xlim(0, 256)
plt.title('B Histogram Car')
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(10,5))
plt.subplot(131)
plt.bar(not_car_bincen, not_car_rh[0])
plt.xlim(0, 256)
plt.title('R Histogram Not-Car')
plt.subplot(132)
plt.bar(not_car_bincen, not_car_gh[0])
plt.xlim(0, 256)
plt.title('G Histogram Not-Car')
plt.subplot(133)
plt.bar(not_car_bincen, not_car_bh[0])
plt.xlim(0, 256)
plt.title('B Histogram Not-Car')
fig.tight_layout()
plt.show()


# ## Feature Extraction: other color spaces
# 
# * Let's see how HOG works for other color spaces than RGB (e.g. HSV, LUV, HLS, YUV, YCrCb)
# * To do that, let's define a new function

# In[29]:

# Define a function to extract features from a list of images
# Have this function call get_hog_features()
def extract_features(img_files, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    spatial_feat = True
    hist_feat = True
    spatial_size=(32, 32)
    hist_bins=32
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for filenum, file in enumerate(img_files):
        file_features = []
        # Read in each one by one
        img = mpimg.imread(file)
        if np.any(img < 0):
            print("An entry in the image is <= 0. @ img=mpimg.imread")
            print(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            else:
                print('Given image space is not defined.')
                print(file)
                print('filenumber: ', filenum)
        else: 
            img = np.uint8(img*255)
            feature_image = np.copy(img)

        if np.any(feature_image < 0):
            print("An entry in the image is <= 0. @ feature_image..")
            print(file)
            print('filenumber: ', filenum)
            
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
            
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# ## Train a linear Support Vector Machine (SVM) classifier
# 
# * Prepare data
# * Train a classifier to detect cars

# In[30]:

# Split the data before shuffling it, that'll help to avoid overfitting
# Explanation: If you shuffle before splitting the dataset into training & validation sets,
# you will have very similar data in both sets, which will set you up for overfitting.
print('len(car_paths): ', len(car_paths))
print('len(not_car_paths): ', len(not_car_paths))

num_val_set = int(len(car_paths) * 0.2)
num_train_set = int(len(car_paths) - num_val_set)
print("num_val_set: ", num_val_set)
print("num_train_set: ", num_train_set)

# Take the same number of car and not-car images. Separate
# training and validation set.
train_set_cars = car_paths[:num_train_set]
train_set_not_cars = not_car_paths[:num_train_set]
val_set_cars = car_paths[num_train_set:]
val_set_not_cars = not_car_paths[num_train_set:]

print("")
print("len(train_set_cars): ", len(train_set_cars))
print("len(train_set_not_cars): ", len(train_set_not_cars))
print("len(val_set_cars): ", len(val_set_cars))
print("len(val_set_not_cars): ", len(val_set_not_cars))


# ### Train a classifier with all YCrCb channels, orient=7, pix/cell=8, cell=block=2

# In[ ]:

# HSV, orient: 12, pix_per_cell: 4, cell_per_block: 4 (97344 features)
colorspace = 'YCrCb'
orient = 7
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"

# Extract features from images with and without cars.
car_features_train = extract_features(train_set_cars, cspace=colorspace, orient=orient, 
                                      pix_per_cell=pix_per_cell, 
                                      cell_per_block=cell_per_block, 
                                      hog_channel=hog_channel)
print('done with calculating car-train features')
notcar_features_train = extract_features(train_set_not_cars, cspace=colorspace, orient=orient, 
                                         pix_per_cell=pix_per_cell, 
                                         cell_per_block=cell_per_block, 
                                         hog_channel=hog_channel)
print('done with calculating notcar-train features')
car_features_val = extract_features(val_set_cars, cspace=colorspace, orient=orient, 
                                    pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel)
print('done with calculating car-val features')
notcar_features_val = extract_features(val_set_not_cars, cspace=colorspace, orient=orient, 
                                       pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel)
print('done with calculating notcar-val features')

print('len(car_features_train): ', len(car_features_train))
print('len(notcar_features_train): ', len(notcar_features_train))
print('len(car_features_val): ', len(car_features_val))


# In[ ]:

# Create an array stack of feature vectors
X_train = np.vstack((car_features_train, notcar_features_train)).astype(np.float64)
X_val = np.vstack((car_features_val, notcar_features_val)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X_train)
print('done fitting the Scaler')
# Apply the scaler to X_train and X_val
scaled_X_train = X_scaler.transform(X_train)
scaled_X_val = X_scaler.transform(X_val)
print('done transforming data')

# Define the labels vector
y_train = np.hstack((np.ones(len(car_features_train)), np.zeros(len(notcar_features_train))))
y_val = np.hstack((np.ones(len(car_features_val)), np.zeros(len(notcar_features_val))))
print('done preparing label vector')

# Shuffle the data
X_train_shuffled, y_train_shuffled = shuffle(scaled_X_train, y_train)
X_val_shuffled, y_val_shuffled = shuffle(scaled_X_val, y_val)
print('done splitting data')
print(X_train_shuffled.size)
print(y_train_shuffled.size)
print(X_val_shuffled.size)
print(y_val_shuffled.size)


# ### Train the classifier and test accuracy
# 
# * Use C=3e-6 to train classifier

# In[ ]:

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train_shuffled[0]))
# Use a linear SVC 
svc3 = LinearSVC(C=3e-6)
# Check the training time for the SVC
t=time.time()
svc3.fit(X_train_shuffled, y_train_shuffled)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc3.score(X_val_shuffled, y_val_shuffled), 4))
print('Train Accuracy of SVC = ', round(svc3.score(X_train_shuffled[:1000], 
                                                   y_train_shuffled[:1000]), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc3.predict(X_val_shuffled[0:n_predict]))
print('For these',n_predict, 'labels: ', y_val_shuffled[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
print()


# ### Pickle the classifier and X_scaler for future use.

# In[ ]:

with open("classifier_ycrbc_with_gamma.pickle", "wb") as f:
    pickle.dump(svc3, f)
    
with open("X_scaler_ycrbc_with_gamma.pickle", "wb") as f:
    pickle.dump(X_scaler, f)


# ### Alternatively, load the saved models

# In[35]:

svc3 = pickle.load(open ("classifier_ycrbc_with_gamma.pickle", "rb"))
X_scaler = pickle.load(open ("X_scaler_ycrbc_with_gamma.pickle", "rb"))


# ## Sliding Window Search
# 
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images

# In[46]:

# Function that draws boxes onto image, based on a list of boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    else:
        print('you have to specify a known conversion')
        return None

# find_cars is able to both extract features & make predictions
# find_cars only has to extract hog features once and then can be 
# sub-sampled to get all of its overlaying windows

# Define a single function that can extract features using hog 
# sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, cells_per_step=2):

    spatial_size=(32, 32)
    hist_bins=32
    box_list = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    # NOTE: If you read in jpg with mpimg, you'll get 0-255 images. 
    # If you read in png, you get 0-1
    # This function wants 0-1, so downscale images with numbers 0-255
    # img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, 
                                     (np.int(imshape[1]/scale), 
                                      np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    # cells_per_step: Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, 
                            feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, 
                            feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, 
                            feature_vec=False)

    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, 
                             xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, 
                             xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, 
                             xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, 
                                                xleft:xleft+window], 
                                (64,64))
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction 
            test_features = X_scaler.transform(np.hstack((spatial_features, 
                                                          hist_features, 
                                                          hog_features)).reshape(1, -1))    

            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            # print(svc.predict_proba())
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
            # if True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append(((xbox_left, ytop_draw+ystart),
                                 (xbox_left+win_draw,ytop_draw+win_draw+ystart)))

                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw,ytop_draw+win_draw+ystart),
                              (0,0,255),6) 
                
    return box_list, draw_img


# ## Test algorithm on example images

# In[47]:

# Test the algorithm on an image
colorspace = 'YCrCb'
orient = 7
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"

box_list = []

# Take one example image from test_images
test_files = glob.glob('test_images/test*.jpg')
test_image = mpimg.imread(test_files[0])

ystart = [380, 380, 400, 400]
ystop = [650, 656, 650, 650]
scale = [2.3, 1.9, 1.4, 1]
# cells_per_step = [9, 6, 8, 3]
cells_per_step = [1, 1, 2, 2]

box_list = []

test_files = [test_files[0]]
for file in test_files:
    test_image = mpimg.imread(file)
    for i in range(len(scale)):
        if (i%2 == 0):
            plt.figure(figsize=(20,20))
        plt.subplot(1,2,i%2+1)
        # Let's create a searching method that best scans the desired region
        boxes, out_img = find_cars(test_image, ystart[i], ystop[i], scale[i], 
                                   svc3, X_scaler, orient, pix_per_cell, 
                                   cell_per_block, cells_per_step[i])
        title = "scale=" + str(scale[i]) + ", cells per step=" + str(cells_per_step[i])

        box_list.extend(boxes)
        plt.imshow(out_img)
        plt.title(title, fontsize=20)
        if (i%2 == 1):
            plt.show()


# In[39]:

# Let's combine all the above boxes into one image
boxes_image = test_image
for box in box_list:
    cv2.rectangle(boxes_image,box[0], box[1], (0,0,255),6) 
plt.imshow(boxes_image)
plt.title('All scales combined')
plt.show()


# ## Add heat method to extract cars from image & finish pipeline

# In[40]:

heat_image = mpimg.imread(test_files[0])
heat = np.zeros_like(heat_image[:,:,0]).astype(np.float)

# Take a list of boxes, and add heat to the heatmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
    
# Apply a threshold to the heatmap
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Draw the labeled boxes in an image
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Add heat to each box in box list
heat = add_heat(heat,box_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(heat_image), labels)


fig = plt.figure(figsize=(20,20))
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()


# In[41]:

def pipeline(img, classifier, X_scaler, heatmap_threshold=2, draw_images=False):
    colorspace = 'YCrCb'
    orient = 7
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"

    test_files = glob.glob('test_images/test*.jpg')
    test_image = mpimg.imread(test_files[0])

    ystart = [380, 380, 400, 400]
    ystop = [650, 656, 650, 650]
    scale = [2.3, 1.9, 1.4, 1]
    cells_per_step = [1, 1, 2, 2]

    box_list = []
    
    for i in range(len(ystart)):
        if (i%2 == 0) and draw_images:
            plt.figure(figsize=(20,20))
        plt.subplot(1,2,i%2+1)
        # Let's create a searching method that best scans the desired region
        boxes, out_img = find_cars(img, ystart[i], ystop[i], scale[i], 
                                   classifier, X_scaler, orient, pix_per_cell, 
                                   cell_per_block, cells_per_step[i])

        box_list.extend(boxes)
        if draw_images:
            plt.imshow(out_img)
            title = "scale=" + str(scale[i]) + ", cells per step=" + str(cells_per_step[i])
            plt.title(title, fontsize=20)
            if (i%2 == 1):
                plt.show()

    if draw_images:
        boxes_image = np.copy(img)
        for box in box_list:
            cv2.rectangle(boxes_image,box[0], box[1], (0,0,255),6) 
        plt.imshow(boxes_image)
        plt.title('All scales combined')
        plt.show()

    heat = np.zeros_like(heat_image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heatmap_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_orig_img = draw_labeled_bboxes(np.copy(img), labels)


    if draw_images:
        fig = plt.figure(figsize=(20,20))
        plt.subplot(121)
        plt.imshow(draw_orig_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()
    
    return heat


# Let's test this on more images!

# In[42]:

test_files = glob.glob('test_images/test*.jpg')
for file in test_files:
    test_image = mpimg.imread(file)

    pipeline(test_image, svc3, X_scaler, heatmap_threshold=1, draw_images=True)


# ## Apply pipeline on a video stream
# 
# * Run your pipeline on a video stream
# * Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.

# In[43]:

heat_list = []
heat_sum = np.zeros((720, 1280)).astype(np.float64)
reached_length = False

# Define a function that can be fed into the video processor
def car_finding(img):
    # Get the heat from an image
    heat = pipeline(img, svc3, X_scaler, heatmap_threshold=2)
    
    # Add the new heat to the heat_sum, and to the heat_list
    # Those keep track of the current sum of heat, and the heat from the past
    # n images.
    global heat_sum, heat_list
    heat_sum += heat
    heat_list.append(heat)
    
    # If the length of the heatlist hits 12, set reached_length = True
    # This is used to make sure that cars detected in the start of the video
    # don't get cropped out because the summed heat is too low
    length_heatlist = len(heat_list)
    if length_heatlist == 12:
        global reached_length
        reached_length = True
        
    # Once we have at least 12 frames, set the heatmap_threshold to 12.
    # If not, set it to the length of heat_list (equivalent to number of
    # processed frames.)
    if reached_length:
        heatmap_threshold = 12
    else:
        heatmap_threshold = length_heatlist
    
    # Here we make sure that the heat_list keeps track of the heat form the
    # past 12 frames.
    if len(heat_list) > 12:
        old_heat = heat_list.pop(0)
        heat_sum -= old_heat
    
    # Visualize the heatmap when displaying  
    heatmap = np.clip(heat_sum, 0, 255)
    
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, heatmap_threshold)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_orig_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return draw_orig_img


# In[ ]:

heat_list = []
heat_sum = np.zeros((720, 1280)).astype(np.float64)

# clip = VideoFileClip('test_video.mp4').subclip(0,1)
clip = VideoFileClip('project_video.mp4').subclip(0, 15)
processed_vid = clip.fl_image(car_finding)
get_ipython().magic("time processed_vid.write_videofile('processed_video.mp4', audio=False)")

