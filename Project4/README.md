# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./readme_images/chessboard_undistortion.png "Chessboard undistorted"
[image2]: ./readme_images/undistortion.png "Undistortion"
[image3]: ./readme_images/warped.png "Warped image"
[image4]: ./readme_images/rg_thresholds.png "R/G combined thresholds"
[image5]: ./readme_images/sobel_thresholds.png "Sobel combined thresholds"
[image6]: ./readme_images/color_thresholds.png "Color combined thresholds"
[image7]: ./readme_images/combined_thresholds.png "Combined all thresholds"
[image8]: ./readme_images/slide_window_search.png "Combined all thresholds"
[image9]: ./readme_images/processed.png "Processed"
[image10]: ./readme_images/final.png "Processed final image"
[video1]: ./project_output.mp4 "Video"

### Submitted files:
1. camera_cal.py (For camera calibration)
2. thresholds.py (Include all the sobel or color space thresholds used for this project)
3. search_lane.py (Slide window search or target search with history tracking, find_lane.py is the file contains all no history tracking searchs)
4. pipeline.py (combined all the image process needed)
5. video_process.py (Use for processing the video with pipeline)
6. line.py (Include Line class for tracking all the infomation need from previous frames)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `camera_cal.py` (in lines 9 through 35 of the file).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result(in method `cal_undistort()` from line 38 to 41): 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, here is the example image showing the distortion correction applied on the lane image(check `pipeline.py` in line 24)
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 46 through 54 in the file `camer_cal.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner(`pipeline.py` line 28 to 29):

```python
src = np.float32([[690, 450], 
				[1110, undist.shape[0]], 
				[220, undist.shape[0]], [
				600, 450]])
dst = np.float32([[1040, 0], 
				[1040, undist.shape[0]], 
				[180, undist.shape[0]], 
				[180, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 690, 450      | 1040, 0        | 
| 1110, 720      | 1040, 720      |
| 220, 720     | 180, 720      |
| 600, 450      | 180, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 39 through 69 in `pipeline.py`, check the thresholds method details in `thresholds.py`).  Here's an example of my output for this step.  (This is from one of the test images)

First, I applied r/g combined thresholds(`pipeline.py` line 48-51):
`combined_binary_gr[(g_binary == 1) & (r_binary == 1)] = 1`

![alt text][image4]
Second, I applied Sobel combined thresholds('pipeline.py' line 55-57):

```python
combined_binary_xmd[((gradx == 1) & (grady == 1)) 
| ((mag_binary == 1) & (dir_binary == 1))] = 1
```
![alt text][image5]
Third, I applied Color space combined thresholds('pipeline.py' line 60-62):
`
combined_binary_sh[(h_binary == 1) | (s_binary == 1)] = 1
`
![alt text][image6]

The total combined shresholds is(`pipeline.py` line 65-67):

```python
combined_binary[((combined_binary_gr == 1) 
& (combined_binary_sh == 1)) 
| (combined_binary_xmd == 1)] = 1
```
![alt text][image7]
#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used the slide window search and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
In line 69 to 82 of `pipeline.py`, I first try to use target search(method find_lane_target_search in `search_lane.py` line 106 to 142)check if this frame can detected lanes or not, if target search can not detect lanes, I'll try to use slide window search to detect the lanes(method find_lane_slide_window in `search_lane.py` line 14 to 103). Then I'll check if the detected lanes have valid distance between left and right lanes and see if left and right curature difference is valid(`pipeline.py` in line 74-82) to decide if we use the previous fit or avg curature.

For the details of the curvature:
I calulated the radius in lines 53 through 56 in cal_radius method of `line.py` , formula used:
`f(y) = Ay^2+By+C`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 84 through 107 in my code in `pipeline.py` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.

![alt text][video1]

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issue: 
Try my model on challenge videos, my pipeline can not find the lanes if cross the bridge or the video hit extreme lights and very sharp curve.
To make it more robust:
1. If detected the lanes for current image, I'll always use current image fit. Otherwise, try to use the avg fit for previous 5 iterations(5 previous fits)
2. Try to experiment on the thresholds combination more. So it can handle more extreme conditions(shadow, extreme lights)

