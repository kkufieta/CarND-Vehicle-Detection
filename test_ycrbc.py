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
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

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

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

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



# ### Alternatively, load the saved models

# In[120]:

svc3 = pickle.load(open ("classifier_ycrbc_with_gamma.pickle", "rb"))
X_scaler = pickle.load(open ("X_scaler_ycrbc_with_gamma.pickle", "rb"))


# ## Sliding Window Search
# 
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images

# In[181]:

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



# In[185]:

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
        # Let's create a searching method that best scans the desired region
        boxes, out_img = find_cars(img, ystart[i], ystop[i], scale[i], 
                                   classifier, X_scaler, orient, pix_per_cell, 
                                   cell_per_block, cells_per_step[i])

        box_list.extend(boxes)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heatmap_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_orig_img = draw_labeled_bboxes(np.copy(img), labels)
    
    return heat



# ## Apply pipeline on a video stream
# 
# * Run your pipeline on a video stream
# * Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.

# In[187]:

heat_list = []
heat_sum = np.zeros((720, 1280)).astype(np.float64)
reached_length = False


def car_finding(img):
    heat = pipeline(img, svc3, X_scaler, heatmap_threshold=2)
    
    global heat_sum, heat_list
    heat_sum += heat
    heat_list.append(heat)
    
    length_heatlist = len(heat_list)
    if length_heatlist == 12:
        global reached_length
        reached_length = True
        
    if reached_length:
        heatmap_threshold = 12
    else:
        heatmap_threshold = length_heatlist
        
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
clip = VideoFileClip('project_video.mp4')
processed_vid = clip.fl_image(car_finding)
processed_vid.write_videofile('processed_video.mp4', audio=False)

