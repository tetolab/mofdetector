import pandas as pd
import numpy as np
import tensorflow as tf


def get_labels(dataframe):
    return dataframe.iloc[:, [0]]


def get_image_data(dataframe):
    return dataframe.iloc[:, 1:785]


def convert_image_df_to_numpy(image_dataframe):
    return image_dataframe.values.reshape((-1, 28, 28))


# the train X data needs to be converted from a numpy array of 64 bit
# integers in the 0-255 range, to a format that tensorflow understands;
# a tensor of 32 bit floats in the 0.0-1.0 range.
# Fortunately tensorflow has a convenient helper function for this:
def convert_image_to_tf_whc(image):
    image = tf.image.convert_image_dtype([image], dtype=tf.float32)
    return tf.reshape(image, (-1, 28, 28, 1))


def flip_and_rotate(image):
    image = np.fliplr(image)
    return np.rot90(image, axes=(2, 1))


def convert_labels(labels):
    y = np.asarray(labels)
    y = np.piecewise(y, [(y != 6) & (y != 13), y == 6, y == 13], [0, 1, 2])
    y = tf.one_hot(y, 3, dtype=tf.int32)
    return tf.reshape(y, (-1, 3))


def get_training_data():
    dataframe = get_data_as_dataframe()
    return get_data(dataframe)


def get_test_data():
    dataframe = get_test_data_as_dataframe()
    return get_data(dataframe)


def get_data(dataframe):
    labels = get_labels(dataframe)
    labels = convert_labels(labels)
    image_data = get_image_data(dataframe)
    image_data = convert_image_df_to_numpy(image_data)
    image_data = flip_and_rotate(image_data)
    image_data = convert_image_to_tf_whc(image_data)
    return image_data, labels


def get_data_as_dataframe():
    return pd.read_csv("data/emnist-letters-train.csv")


def get_test_data_as_dataframe():
    return pd.read_csv("data/emnist-letters-test.csv")
