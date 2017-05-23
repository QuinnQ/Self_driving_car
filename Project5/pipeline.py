from classifer import *
from find_car_boxes import *
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

format = 'png'
car_path = './data/vehicles'
notcar_path = './data/non-vehicles'
color_space = 'RGB2YCrCb'
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()
ystart = 400
ystop = 670
scale_st = 1.0
scale_end = 2.1
threshold = 2

history_boxes = []

# Train the classifer
# svc, X_scaler = train_classifer(car_path, notcar_path, format, color_space, spatial_size, hist_bins, orient,
#                                 pix_per_cell, cell_per_block, hog_channel, spatial_feat,
#                                 hist_feat, hog_feat)
#

# Load saved model
svc, X_scaler = joblib.load('svc.pkl'), joblib.load('scaler.pkl')


def pipeline(img):
    bboxes=[]
    threshold = 0
    for scale in np.arange(scale_st, scale_end, 0.5):
        bbox = find_cars(img, ystart, ystop, scale, svc, X_scaler,
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        bboxes.extend(bbox)
        threshold += 1
    filtered_boxes, heatmap = filter_bboxes(img, bboxes, threshold)

    # if len(history_boxes) > 10:
    #     history_boxes.pop(0)
    #
    # history_boxes.append(filtered_boxes)
    # flatten = []
    # [flatten.extend(el) for el in history_boxes]
    # box_list, frame_heatmap = filter_bboxes(img, flatten, 1)
    draw_img = draw_boxes(img, filtered_boxes)
    return draw_img

test_img = mpimg.imread('./test_images/test6.jpg')
draw_img = pipeline(test_img)
plt.imshow(draw_img)
plt.show()

# output = 'project5_output.mp4'
# clip = VideoFileClip('project_output.mp4')
# output_clip = clip.fl_image(pipeline)
# output_clip.write_videofile(output, audio=False)


# def visualize_hist(img1, nbins=32):
#     img = convert_color(img1, 'RGB2HLS')
#     channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
#     channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
#     channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
#     bin_edges = channel1_hist[1]
#     bin_centers = (bin_edges[1:]+bin_edges[0:len(bin_edges)-1])/2
#     fig=plt.figure(figsize=(12,4))
#     plt.subplot(141)
#     plt.imshow(img1)
#     plt.subplot(142)
#     plt.bar(bin_centers, channel1_hist[0])
#     plt.xlim(0, 256)
#     plt.title('H Histogram')
#     plt.subplot(143)
#     plt.bar(bin_centers, channel2_hist[0])
#     plt.xlim(0, 256)
#     plt.title('L Histogram')
#     plt.subplot(144)
#     plt.bar(bin_centers, channel3_hist[0])
#     plt.xlim(0, 256)
#     plt.title('S Histogram')
#     fig.tight_layout()
#     plt.show()
#
# cars, notcars = load_data(car_path, notcar_path, format)
# rand1 = np.random.choice(cars)
# # rand2 = np.random.choice(notcars22
# img1 = mpimg.imread(rand1)
# # feature_image = convert_color(img1, color_space)
# # img2 = cv2.resize(feature_image[:, :, 0], (32, 32))
# # features, hog_image = get_hog_feature(feature_image[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
# # display_images(img1, img2, 'Vehicle', 'Binned Color Features')
# visualize_hist(img1)

