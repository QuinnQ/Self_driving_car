# **Finding Lane Lines on the Road**

### **Goals**
* Make a pipeline that finds lane lines on the road
* Apply the pipeline to images and a video stream
* Practice and use tools provided from opencv.
* The brief steps will be:
   1. Use grayscale, gaussian_blur, canny to reduce the noise and make mask for images.
   2. Convert masked image from image space to hough space and use different parameters to filter and get hough lines.
   3. The final step will be draw the image and save the image.

### **Steps and Changes I Made**

#### The steps of this project are the following:
* Change the image to be gray scale
* Use Gussian smoothing, to reduce image noise and reduce details
* Use Canny Algorithm get the edge of the image
* Make Polygon mask with vertical ratio = 7/12 and horizontal ratio 5/8
* Convert masked image to hough line space and filter all the lines with parameters(rho = 2, theta = pi/180, threshold = 10, min_line_len = 20 and max_line_gap = 10)
* I modified draw line function:
    1. Keep track of right lane slopes and left lane slopes
    2. Keep track of right lane lines and left lane lines
    3. Apple a slope filter to filter out the noise lines
    4. Always use the average of the left slope/right slope
    5. Find the left/right top point with min y value(one of them will not have min y, update later)
    6. Calculate the left/right bottom point base on slopes and top points
    7. Now update the top point which don't have min y base on the bottom point and slope.
    8. Also use 2 global variables keep tracking the previous top points. In case that we don't get valid lines.
    9. Then draw the left lane and right lane
* Make a pipeline that finds lane lines on the road base on previous steps
* Save the image, and here is the example output I got:
[image1]: ./test_images/solidYellowCurve2_with_lane.jpg "Example output"
![alt text][image1]
---

### **Potential Shortcomings**

#### 1. Shortcomings:
This method still not robost enough to deal with images with sharp turn, this kind of lines will filter out by the slope filter. And if the image have bridge-like structures along with the road, the method also will confused, cause hough line will not filter that out.

#### 2. Possible improvements:
Image normally include 2 types of informations: edges and colors. For the hough line we are making use of the edges. For the colors, besides of using polygon mask(area restrictions), we can apply color masks to the images to filter out the yellow and white lanes(color restrictions).
To make extrapolate line working better, we can also keep track of the average slopes and average intercepts. Especially for the vedio, history of the slopes and intercepts will have us to reduce noise and make decision when we short of current image information.
