import numpy as np
import cv2
import matplotlib.pyplot as plt

slide_window_n = 9
margin = 100
minpixel = 50
ym_per_pix = 30/720
xm_per_pix = 3.7/700


def find_lane_slide_window(img, print=False):
    # Take bottom half of the image
    half = np.int(img.shape[0] / 2)
    histogram = np.sum(img[half:, :], axis=0)
    # Create an output image to draw on and visualize the result
    if print:
        out_img = np.dstack((img, img, img)) * 255
    # Find the peak fo left and right
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:])+midpoint
    # window height
    window_h = np.int(img.shape[0]/slide_window_n)
    # Identify the x and y positions of all non zero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = left_base
    rightx_current = right_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_indx = []
    right_lane_indx = []
    # Step through window one by one
    for window in range(slide_window_n):
        win_y_low = img.shape[0] - (window+1) * window_h
        win_y_high = img.shape[0] - window * window_h
        win_leftx_low = leftx_current - margin
        win_leftx_high = leftx_current + margin
        win_rightx_low = rightx_current - margin
        win_rightx_high = rightx_current + margin
        # Draw window on visualization image
        if print:
            cv2.rectangle(out_img, (win_leftx_low, win_y_low), (win_leftx_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_rightx_low, win_y_low), (win_rightx_high, win_y_high), (0, 255, 0), 2)
        # Identify nonzero pixels in x and y within window
        good_left_indx = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                          & (nonzerox >= win_leftx_low) & (nonzerox <= win_leftx_high)).nonzero()[0]
        good_right_indx = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                          & (nonzerox >= win_rightx_low) & (nonzerox <= win_rightx_high)).nonzero()[0]

        left_lane_indx.append(good_left_indx)
        right_lane_indx.append(good_right_indx)
        # If you found  > minpixel, recenter next window
        if len(good_left_indx) > minpixel:
            leftx_current = np.int(np.mean(nonzerox[good_left_indx]))
        #     left_lane_indx.append(good_left_indx[np.random.randint(0, len(good_left_indx), minpixel)])
        # else:
        #     left_lane_indx.append(good_left_indx)
        if len(good_right_indx) > minpixel:
            rightx_current = np.int(np.mean(nonzerox[good_right_indx]))
        #     right_lane_indx.append(good_right_indx[np.random.randint(0, len(good_right_indx), minpixel)])
        # else:
        #     right_lane_indx.append(good_right_indx)

    # Concatenate the arrays of indices
    left_lane_indx = np.concatenate(left_lane_indx)
    right_lane_indx = np.concatenate(right_lane_indx)

    left_curv, right_curv = measure_curvature(img, nonzerox, nonzeroy, left_lane_indx, right_lane_indx)

    if print:
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_indx]
        lefty = nonzeroy[left_lane_indx]
        rightx = nonzerox[right_lane_indx]
        righty = nonzeroy[right_lane_indx]
        # Fit a second order polynomial to left and right(here depends on y find x)
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Visualization
        ploty = np.linspace(0, img.shape[0]-1, num=img.shape[0])
        left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]

        # Color in left and right lanes
        out_img[nonzeroy[left_lane_indx], nonzerox[left_lane_indx]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indx], nonzerox[right_lane_indx]] = [255, 0, 0]
        cv2.putText(out_img, 'left: %.2f m, right: %.2f m' % (left_curv, right_curv),
                    (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    return nonzerox, nonzeroy, left_lane_indx, right_lane_indx


def find_lane_target_search(img, nonzerox, nonzeroy, left_lane_indx, right_lane_indx):
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx]
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]
    # Fit a second order polynomial to left and right(here depends on y find x)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Identify the x and y positions of all non zero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Get left and right indx
    leftx_low = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin
    leftx_high = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin
    rightx_low = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin
    rightx_high = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin

    left_lane_indx = ((nonzerox >= leftx_low) & (nonzerox <= leftx_high))
    right_lane_indx = ((nonzerox >= rightx_low) & (nonzerox <= rightx_high))

    return nonzerox, nonzeroy, left_lane_indx, right_lane_indx


def measure_curvature(img, nonzerox, nonzeroy, left_lane_indx, right_lane_indx):
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx]
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]
    # Fit a second order polynomial to left and right(here depends on y find x)
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = img.shape[0]-1
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
    return left_curverad, right_curverad


def visualization(img, nonzerox, nonzeroy, left_lane_indx, right_lane_indx):
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_indx], nonzerox[left_lane_indx]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_indx], nonzerox[right_lane_indx]] = [0, 0, 255]

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx]
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]
    # Fit a second order polynomial to left and right(here depends on y find x)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualization
    ploty = np.linspace(0, img.shape[0] - 1, num=img.shape[0])
    left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img.shape[0] - 1, num=img.shape[0])
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    left_curv, right_curv = measure_curvature(img, nonzerox, nonzeroy, left_lane_indx, right_lane_indx)
    cv2.putText(out_img, 'left: %.2f m, right: %.2f m' % (left_curv, right_curv),
                (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

