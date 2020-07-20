import tensorflow as tf

import csv
import os

import gan_config as config

# Create config:
cfg = config.Config()


def create_dataset(train):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    ds = tf.data.Dataset.from_tensor_slices(train_images)

    if train:
        ds = ds.shuffle(cfg.buffer_size)

    return ds.batch(cfg.ds_batch)




