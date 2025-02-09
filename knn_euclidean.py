import numpy as np
import tensorflow as tf


def knn(input_arr, k):
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    flattened_images = tf.reshape(train_images, [train_images.shape[0], -1])

    diffs = flattened_images - input_arr
    sq_diffs = tf.square(diffs)
    sums = tf.reduce_sum(sq_diffs, axis=1)
    distances = tf.sqrt(sums)
    sorted_dist = tf.argsort(distances)

    nearest_labels = [int(train_labels[i]) for i in sorted_dist.numpy()[:k]]
    most_common = np.bincount(nearest_labels).argmax()
    return most_common

