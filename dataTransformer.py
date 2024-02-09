import tensorflow as tf
from model_params import *
import numpy as np


CROP_SHAPE = (96,96)
ROTATION_FACTOR = 0.3
CONTRAST_FACTOR = 0.8


class TransformPipeline:
    def __init__(self, transforms:list):
        self.transforms = transforms
        
    def transform(self,image:tf.Tensor, label):
        for T in self.transforms:
            image = T(image,label)
        return image,label


def crop(image, label):
    
    return tf.image.random_crop(image,size=CROP_SHAPE), label

def contrast(image, label):
    return tf.image.random_contrast(image,1-CONTRAST_FACTOR,1+CONTRAST_FACTOR), label

def flip(image, label):
    return tf.image.random_flip_up_down(image), label


class TransformLayer(tf.keras.Model):
    def __init__(self, layers:list=[], input_shape=INPUT_SHAPE):
        super(TransformLayer,self).__init__(name='transform')
        if len(layers) == 0:
            layers.append(tf.keras.layers.Resizing(input_shape[0],input_shape[1]))
        self.transformations = layers
        
    def call(self, input_tensor, training=False):
        for T in self.transformations:
            input_tensor = T(input_tensor,training=training)
        return input_tensor
    

def create_transformation_layer():
    transforms = []
    transforms.append(tf.keras.layers.Rescaling(1./255))
    transforms.append(tf.keras.layers.CenterCrop(IMG_DIM[0]-10,IMG_DIM[1]-10))
    transforms.append(tf.keras.layers.RandomRotation(ROTATION_FACTOR))
    transforms.append(tf.keras.layers.RandomCrop(CROP_SHAPE[0],CROP_SHAPE[1]))
    transforms.append(tf.keras.layers.Resizing(INPUT_SHAPE[0],INPUT_SHAPE[1]))
    transforms.append(tf.keras.layers.RandomContrast(CONTRAST_FACTOR))
    
    return transforms 

class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filt, name=None):
        super(ResnetBlock, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.filt = filt
        

        self.conv2a = tf.keras.layers.Conv2D(filt, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filt, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filt*2, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

        

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.keras.layers.Activation('relu')(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x = x + input_tensor

        return tf.keras.layers.Activation('relu')(x)
        
    def get_config(self):
        return {"kernel_size": self.kernel_size, "filt":self.filt}



def create_model(num_classes,transformation):
    model = tf.keras.Sequential([
        *transformation,
        #tf.keras.layers.Conv2D(32, 7, activation='relu',padding='same'),
        #ResnetBlock(7,16,'res1'),
        tf.keras.layers.Conv2D(64, 5, activation='relu', padding='same',strides=2),#size/2
        ResnetBlock(5,32,'res2'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',strides=2),#size/4
        ResnetBlock(3,64,'res3'),
        tf.keras.layers.Conv2D(256, 1, activation='relu', padding='same',strides=2),#size/8
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256*8,activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model(tf.ones((1,*IMG_DIM)))
    return model



