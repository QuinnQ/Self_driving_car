import cv2
from skimage.feature import hog
import numpy as np
import matplotlib.image as mpimg


def convert_color(img, conv="RGB2YCrCb"):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)


def get_hog_feature(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def hist_color(img, nbins=32, channels=(0, 1, 2)):
    hist_features = []
    for channel in channels:
        channel_hist = np.histogram(img[:, :, channel], bins=nbins)
        hist_features = np.concatenate(channel_hist)
    return hist_features


def img_extract_features(img, color_space, spatial_size=(32, 32), hist_bins=32, orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
                         hist_feat=True, hog_feat=True):
    features = []
    feature_image = convert_color(img, color_space)
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        features.append(spatial_features)
    if hist_feat:
        # Apply color_hist()
        image = convert_color(img, 'RGB2HLS')
        hist_features = hist_color(image, nbins=hist_bins, channels=(0, 2))
        features.append(hist_features)
    if hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_feature(feature_image[:, :, channel],
                                                    orient, pix_per_cell, cell_per_block,
                                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_feature(feature_image[:, :, hog_channel], orient,
                                           pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        features.append(hog_features)
    return features


def extract_features(img_files, color_space, spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
                     hist_feat=True, hog_feat=True):
    combined_features = []
    for img_file in img_files:
        img = mpimg.imread(img_file)
        features = img_extract_features(img, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                        cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        combined_features.append(np.concatenate(features))
    return combined_features
