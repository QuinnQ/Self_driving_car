import numpy as np
import cv2
import matplotlib.pyplot as plt


def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with OpenCV
    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = (255*abs_sobel/np.max(abs_sobel)).astype(np.uint8)
    # Create a copy and apply threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel <= thresh[1]) & (scaled_sobel >= thresh[0])] = 1
    return binary_output


def magnitude_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with OpenCV
    abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    gradmag = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a copy and apply threshold
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag <= thresh[1]) & (gradmag >= thresh[0])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply x or y gradient with OpenCV
    abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # Create a copy and apply threshold
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir <= thresh[1]) & (absgraddir >= thresh[0])] = 1
    return binary_output


# HLS images (Hue, Saturation, Lightness), normally use H/S, H(0-179), S(0-255)
def hsv_color_threshold(channel, thresh=(0, 255)):
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


# RGB images normally use R/G, R(200-255), G(180-255)
def rgb_color_threshold(img, ch='r', thresh=(180, 255)):
    channel = img[:, :, 0]  # R channel
    if ch == 'g':
        channel = img[:, :, 1]  # G channel
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def stack_channels(ch1, ch2, ch3=None):
    # np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.dstack((np.zeros_like(ch1), ch1, ch2))
    if ch3 is not None:
        color_binary[:, :, 0] = ch3
    return color_binary


def visualization(original, modified, title1, title2, isgray=True):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title(title1, fontsize=20)
    # cv2.putText(modified, pltname,
    #             (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    if isgray:
        ax2.imshow(modified, cmap='gray')
    else:
        ax2.imshow(modified)
    ax2.set_title(title2, fontsize=20)
    plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0)
