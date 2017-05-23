
# **Vehicle Detection Project**

## The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./readme_images/vehicle_non_vehicle.png
[image2]: ./readme_images/HOG.png
[image3]: ./readme_images/binned.png
[image4]: ./readme_images/histogram.png
[image5]: ./readme_images/frame1.png
[image6]: ./readme_images/frame2.png
[image7]: ./readme_images/frame3.png
[image8]: ./readme_images/frame4.png
[image9]: ./readme_images/frame5.png
[image10]: ./readme_images/frame6.png
[image11]: ./readme_images/combined_6_frames.png
[image12]: ./readme_images/filter_frames.png
[image13]: ./readme_images/output_ex1.png
[image14]: ./readme_images/output_ex2.png
[image15]: ./readme_images/output_ex3.png
[video1]: ./project_output_video.mp4


## Files included and details:
1. **`feature_extract.py`** This file include all the details about features extraction(binned color features, histograms of color features, HOG features).
2. **`classifer.py`** This file include all the details about trainning classifier and generate scaler and model.
3. **`find_car_boxes.py`** This file include implementation of a sliding_window technique to search for vehicles. It also includes a heatmap for filtering the outliners.
4. **`pipeline.py`** This file include a pipeline on a video stream.



## Feature Selections
### Histogram of Oriented Gradients (HOG)

#### 1. How to extracted HOG features from the training images.

The code for this step is contained in line 18-32 `feature_extract.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is the final choice of HOG parameters using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, and here is an example image:

![alt text][image2]


#### 2. Use the color features(binned color features and histogram color features) with HOG.
From line 55-62 in `feature_extract.py`:

For binned Color features, parameter used `spatial_size = (32, 32)`, and here is the example image for `YCrCb` color space and color channel Y:
![alt text][image3]
Use `ravel()` method to convert the features to vectors.

For histogram color features, parameter using the `HLS` color space and `nbins = 32`, and here is the example image, and picked channel H and S color features:
![alt text][image4]

#### 3. Trained a classifier using selected HOG features and color features.

I trained a linear SVM with following steps(Line 30-68 in `classifer.py` in method `train_classifer`):

1. Load image data paths to `cars` and `notcars` array
2. For cars/not_cars image files, extract features using concatenated HOG features and color features(Line 79-88 in `extract_features` method of `feature_extract.py`)
3. Use `sklearn.preprocessing.StandardScaler` to normailize all the features
4. Use `sklearn.model_selection.train_test_split` to split data to be training and testing set.
5. Use `sklearn.svm.LinearSVC` to train the training data and save the model to `svc.pkl` and `scaler.pkl`

### Sliding Window Search

#### 1. Implementation details for sliding window search

I decided to search window positions from `ystart=400` and `ystop=670` at `scales=(1.0, 1.5, 2.0)` and instead of usng overlap, defined how many cells to step(`cells_per_step = 2`). 

Compute the feature vectors follow the same parameters/steps as I did for training the classifier. But instead of compute the HOG features step by step in line 40-72 in `find_car_boxes.py`, compute the individual channel HOG features for entire image then extract the features in the for loop.

Then extract features window by window in line 40-62, use the classifier scaler to normalize the combined features. 

Use svc model to predict if there is a car, if there is a car, then append the position to box list(line 65-72)
  

#### 2. Some examples of test images after the processing of pipeline. 

![alt text][image13]
![alt text][image14]
![alt text][image15]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
![alt text][video1]
Here's a [link to my video result](./project_output_video.mp4)


#### 2. Implemented filter for false positives and combining overlapping bounding boxes.

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected(line 99-106 in `find_car_boxes.py`). 

Make history boxes tracking list which contains previous frames filtered boxes(default set = 10 frames), use method filter_bboxes_history in line 109-115 of `find_car_boxes.py` to get the boxes covered area for current image.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]



---

### Discussion

#### 1. Discuss any problems / issues you faced in your implementation of this project. What could you do to make it more robust?

The issue I faced in the implementation is that multiple boxes can be show in one car, to make it more robust, I may need to try use more history tracking data, or use a factor parameteres to group boxes.

