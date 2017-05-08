import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import thresholds as threshs
import matplotlib.pyplot as plt

cal_img_shape = mpimg.imread('./camera_cal/calibration1.jpg').shape


def camera_calibrate(files_path, nx=9, ny=6):

    # Read in and make a list of calibration images
    images = glob.glob(files_path)

    # Array to store object points and image points
    obj_points = []
    img_points = []

    # Prepare object points
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for frame in images:
        img = cv2.imread(frame)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If corners found, add object points and image points
        if ret:
            img_points.append(corners)
            obj_points.append(objp)
            # Draw corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)
    return obj_points, img_points


def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cal_img_shape[0:2], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def warp(img, src, dst):
    img_size = (img.shape[1], img.shape[0])  # w, h
    # Compute the perspective transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective Minv
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Create warped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv


def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        print(event.xdata, event.ydata)

# original = cv2.imread('./camera_cal/calibration1.jpg')
# obj_points, img_points = camera_calibrate('./camera_cal/calibration*.jpg')
# undist = cal_undistort(original, obj_points, img_points)
#
# threshs.visualization(original, undist, 'Original', 'Distortion corrected')
# plt.show()
