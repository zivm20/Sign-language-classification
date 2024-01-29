import tensorflow as tf


def preProcessing(img):
    return tf.data.Dataset.from_tensor_slices(img)

