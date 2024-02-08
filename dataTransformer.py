import tensorflow as tf


def transformPipelline(image, label):
    image = tf.image.random_crop()
    