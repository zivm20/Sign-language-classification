import tensorflow as tf
import cv2
import numpy as np


BATCH_SIZE = 32
IMG_DIM = (128,128,3)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
INPUT_SHAPE = (112,112,3)
TRAIN_VAL_SPLIT = (0.6,0.2)
CACHE_DIR = "cache"
CHECKPOINT_FILEPATH="checkpoints"
BACKUP_FILEPATH="backups"
MODEL_DIR = "models"
ROTATION_FACTOR = 0.3
CONTRAST_FACTOR = 0.8


        

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape != IMG_DIM:
        img = cv2.resize(img, dsize=(IMG_DIM[0],IMG_DIM[1]))
    img = np.asarray(img)
    img = tf.expand_dims(img,0)
    img = tf.image.resize(img, (int(INPUT_SHAPE[0]), int(INPUT_SHAPE[1])))/255.0

    
    return img




def transform_box_T(box,original_image):
    newImg =cv2.resize(original_image, (INPUT_SHAPE[0]*2,INPUT_SHAPE[1]*2))
    box = box * tf.constant([INPUT_SHAPE[1]*2, INPUT_SHAPE[0]*2, INPUT_SHAPE[1]*2, INPUT_SHAPE[0]*2],dtype=tf.float32)
    box = tf.cast(box, tf.int32)
    newImg = cv2.rectangle(newImg,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
    
    return newImg, box




