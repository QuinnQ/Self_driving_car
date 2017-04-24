import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Input
import matplotlib.pyplot as plt
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('network', 'nvidia', "The model to bottleneck, one of 'nvidia', 'vgg', 'inception', or 'resnet'")
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_string('root_path', './', "Data root path")
flags.DEFINE_boolean('augmentation', False, "Augmentation apply on training data")
flags.DEFINE_boolean('include_counter_clock', False, "Have counter clock driving data")
flags.DEFINE_integer('epoches', 5, "Epoch number")

ROOT = FLAGS.root_path
LOG_PATH = ROOT + 'driving_log.csv'
COUNTER_LOG_PATH = ROOT + 'driving_log_counter.csv'
IMG_PATH = ROOT + 'IMG/'
correction = 0.2
epoch = FLAGS.epoches
row, col, ch = 160, 320, 3
b_size = FLAGS.batch_size
augmentation_flag = FLAGS.augmentation
counter_clock_flag = FLAGS.include_counter_clock

h, w, ch2 = 224, 224, 3
if FLAGS.network == 'inception':
    h, w, ch2 = 299, 299, 3
    from keras.applications.inception_v3 import preprocess_input


def generator(samples, size=b_size):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, size):
            batch_samples = samples[offset:offset+size]
            images, angles = [], []
            for batch_sample in batch_samples:
                extract_data(batch_sample, images, angles)
            if augmentation_flag:
                images, angles = data_augmentation(images, angles)
            x_train = np.array(images)
            if FLAGS.network != 'nvidia':
                x_train = preprocess_input(x_train)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


def extract_data(batch_sample, images, angles):
    center_angle = float(batch_sample[3])
    left_angle = center_angle + correction
    right_angle = center_angle + correction
    center_image = get_image_data(batch_sample, 0)
    left_image = get_image_data(batch_sample, 1)
    right_image = get_image_data(batch_sample, 2)
    images.extend([center_image, left_image, right_image])
    angles.extend([center_angle, left_angle, right_angle])


def get_image_data(batch_sample, column_num):
    source_path = batch_sample[column_num]
    filename = source_path.split('/')[-1]
    current_path = IMG_PATH + filename
    img = cv2.imread(current_path)
    return img


def visualize_loss(history):
    print(history.history.keys())
    # plot each epoch with loss, validation_loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE Loss')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def data_augmentation(images, angles):
    augment_images, augment_angles = [], []
    for image, angle in zip(images, angles):
        augment_images.append(image)
        augment_images.append(cv2.flip(image, 1))
        augment_angles.append(angle)
        augment_angles.append(angle*-1.0)
    return augment_images, augment_angles


def create_model():
    input_tensor = Input(shape=(h, w, ch2))
    if FLAGS.network == 'vgg':
        model = VGG16(input_tensor=input_tensor, include_top=False)
    elif FLAGS.network == 'inception':
        model = InceptionV3(input_tensor=input_tensor, include_top=False)
    elif FLAGS.network == 'nesnet':
        model = ResNet50(input_tensor=input_tensor, include_top=False)
    else:
        model = Sequential()
        # data pre-processing
        model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(row, col, ch)))
        model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(row, col, ch)))
        # deep learning steps
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model
    o = model.output
    o = Flatten()(o)
    o = Dense(100)(o)
    o = Dense(50)(o)
    o = Dense(10)(o)
    o = Dense(1)(o)
    model = Model(model.input, o)
    return model


def main(_):
    data = []
    with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(line)

    if counter_clock_flag:
        with open(COUNTER_LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for line in reader:
                data.append(line)
    train_samples, validation_samples = train_test_split(data, test_size=0.2)
    train_generator = generator(train_samples, size=b_size)
    validation_generator = generator(validation_samples, size=b_size)

    model = create_model()

    model.compile(loss='mse', optimizer='adam')
    train_len = len(train_samples)*3
    val_len = len(validation_samples)*3
    if augmentation_flag:
        train_len = train_len * 2
        val_len = val_len * 2
    history = model.fit_generator(train_generator, samples_per_epoch=train_len,
                                  validation_data=validation_generator, nb_val_samples=val_len,
                                  nb_epoch=epoch, verbose=1)
    model.save('model.h5')
    visualize_loss(history)

if __name__ == '__main__':
    tf.app.run()
