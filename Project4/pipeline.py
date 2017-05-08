import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import camera_cal as ccal
import thresholds as threshs
# import find_lanes as fl
import search_lane as sl
from line import Line, xm_per_pix

cal_path = './camera_cal/calibration*.jpg'
img_path = './test_images/test5.jpg'  # straight_lines1, test1!! test4!!
cal_img_shape = mpimg.imread('./camera_cal/calibration1.jpg').shape
img_shape = mpimg.imread(img_path).shape
img_w = img_shape[0]
img_h = img_shape[1]
ksize = 3
objpoints, imgpoints = ccal.camera_calibrate(cal_path)
lines = {'left': Line(), 'right': Line()}


def pipeline(img):
    # undistortion
    undist = ccal.cal_undistort(img, objpoints, imgpoints)
    # threshs.visualization(img, undist, 'Original', 'Distortion corrected')

    # Transfer perspective to bird-view
    src = np.float32([[690, 450], [1110, undist.shape[0]], [220, undist.shape[0]], [600, 450]])
    dst = np.float32([[1040, 0], [1040, undist.shape[0]], [180, undist.shape[0]], [180, 0]])
    warped, Minv = ccal.warp(undist, src, dst)
    # threshs.visualization(undist, warped, 'Distortion corrected', 'Warped')

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:, :, 0]
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # Apply thresholds
    gradx = threshs.abs_sobel_threshold(l_channel, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = threshs.abs_sobel_threshold(l_channel, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = threshs.magnitude_threshold(l_channel, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = threshs.dir_threshold(l_channel, sobel_kernel=15, thresh=(0.7, 1.3))
    s_binary = threshs.hsv_color_threshold(s_channel, thresh=(170, 255))  # 170, 255
    h_binary = threshs.hsv_color_threshold(h_channel, thresh=(20, 100))  # 20, 100

    # rgb thresholds
    r_binary = threshs.rgb_color_threshold(warped, ch='r', thresh=(150, 255))  # 130, 255
    g_binary = threshs.rgb_color_threshold(warped, ch='g', thresh=(150, 255))
    combined_binary_gr = np.zeros_like(r_binary)
    combined_binary_gr[(g_binary == 1) & (r_binary == 1)] = 1

    # threshs.visualization(warped, combined_binary_gr, 'Distortion corrected', 'R/G thresholds')

    # sobel thresholds
    combined_binary_xmd = np.zeros_like(r_binary)
    combined_binary_xmd[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # threshs.visualization(warped, combined_binary_xmd, 'Distortion corrected', 'Sobel thresholds')

    # color thresholds
    combined_binary_sh = np.zeros_like(s_binary)
    combined_binary_sh[(h_binary == 1) | (s_binary == 1)] = 1
    # threshs.visualization(warped, combined_binary_sh, 'Distortion corrected', 'Color thresholds')

    # combine thresholds
    combined_binary = np.zeros_like(combined_binary_gr)
    combined_binary[((combined_binary_gr == 1) & (combined_binary_sh == 1)) | (combined_binary_xmd == 1)] = 1
    # threshs.visualization(warped, combined_binary, 'Distortion corrected', 'Processed image')
    sl.find_lane_target_search(combined_binary, lines)
    detected = lines['left'].detected and lines['right'].detected
    if not detected:
       sl.find_lane_slide_window(combined_binary, lines, print=False)
    detected = lines['left'].detected and lines['right'].detected

    if lines['left'].check_curves_diff(lines['right'])  \
            or (not 400*xm_per_pix < lines['left'].cal_distance(lines['right']) < 750*xm_per_pix):
        for key in lines.keys():
            if lines[key].check_xfitted_diff() < 1000:
                lines[key].update()
    else:
        for key in lines.keys():
            if lines[key].detected:
                lines[key].update()

    # Visualization
    warped_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warped_zero, warped_zero, warped_zero))

    # Extract left and right line pixel positions
    if lines['left'].detected:
        left_fit = lines['left'].current_fit
    else:
        left_fit = lines['left'].avg_fit

    if lines['right'].detected:
        right_fit = lines['right'].current_fit
    else:
        right_fit = lines['right'].avg_fit

    ploty = lines['left'].yfit
    left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank to original image space using inverse perspective matrix
    newWarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)
    # Combine the result with original image
    result = cv2.addWeighted(undist, 1, newWarp, 0.3, 0)

    lane_curvature = np.mean((lines['left'].radius, lines['right'].radius))
    cv2.putText(result, 'Lane curvature: %.2f m' % lane_curvature,
                (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    center = xm_per_pix*640
    cur_car = (lines['right'].base_pos - lines['left'].base_pos)/2 + lines['left'].base_pos
    if cur_car - center > 0:
        cur_pos = 'right'
    else:
        cur_pos = 'left'
    cv2.putText(result, 'Car is in %s of center: %.2f m' %
                (cur_pos, np.absolute(cur_car-center)),
                (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    return result


def main():
    img = mpimg.imread(img_path)
    result = pipeline(img)
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()

