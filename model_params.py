import tensorflow as tf
import cv2
import numpy as np


BATCH_SIZE = 32
IMG_DIM = (90,90)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']



def preProcessing(img):
    newImg = cv2.resize(img, dsize=IMG_DIM)
    newImg = np.asarray(newImg)
    newImg = tf.expand_dims(newImg,0)
    
    return newImg

