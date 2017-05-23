import glob
import os
from feature_extract import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def read_path(root, format):
    images = []
    for path, subdirs, files in os.walk(root):
        for subdir in subdirs:
            target_path = root + '/' + subdir + '/*.' + format
            images.append(glob.glob(target_path))
    return images


# Read in cars and notcars
def load_data(car_path, notcar_path, format):
    car_images = read_path(car_path, format)
    car_images = np.concatenate(car_images)
    non_car_images = read_path(notcar_path, format)
    non_car_images = np.concatenate(non_car_images)
    return car_images, non_car_images


def train_classifer(car_path, notcar_path, format, color_space, spatial_size, hist_bins, orient,
                    pix_per_cell, cell_per_block, hog_channel, spatial_feat,
                    hist_feat, hog_feat):
    cars, notcars = load_data(car_path, notcar_path, format)
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Normalize the features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split data to training and testing sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    svc = LinearSVC()
    # svc = SVC(kernel='rbf', C=1.0, gamma=1000)
    svc.fit(X_train, y_train)
    joblib.dump(svc, 'svc.pkl')
    joblib.dump(X_scaler, 'scaler.pkl')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler


def display_images(img1, img2, title1, title2):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.tight_layout()
        ax1.imshow(img1)
        ax1.set_title(title1, fontsize=20)

        ax2.imshow(img2, cmap='gray')
        ax2.set_title(title2, fontsize=20)
        plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0)
        plt.show()



