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
    def __init__(self, kernel_size, filt, filt_in=None,filt_out=None,num=3,conv_block=False,stride=1,bottleNeck=True, name=None):
        super(ResnetBlock, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.filt = filt
        self.num = num
        self.bottleNck=bottleNeck
        if filt_out == None and filt_in == None:
            filt_out = filt*stride
        elif filt_out == None:
            filt_out = filt_in*stride
        if filt_in != None:
            filt_in = filt

        self.filt_out = filt_out
        self.filt_in = filt_in
        self.conv_layers = []
        self.norm_layers = []
        self.stride = stride

        self.conv_block = (self.filt_out != self.filt_in or stride>1 or conv_block)

        if bottleNeck == True and num>2:
            self.conv_layers.append(tf.keras.layers.Conv2D(self.filt, 1,strides=stride))
        else:
            self.conv_layers.append(tf.keras.layers.Conv2D(self.filt, kernel_size,padding='same',strides=stride))
        self.norm_layers.append(tf.keras.layers.BatchNormalization())
        
        
        for i in range(1, num-1):
            if bottleNeck == True and (i!=num//2 and i!=(num-1)//2):
               self.conv_layers.append(tf.keras.layers.Conv2D(self.filt, 1))
            else:
                self.conv_layers.append(tf.keras.layers.Conv2D(self.filt, kernel_size, padding='same'))
            self.norm_layers.append(tf.keras.layers.BatchNormalization())
            

        if bottleNeck == True and num>2:
            self.conv_layers.append(tf.keras.layers.Conv2D(self.filt_out, 1))
        else:
            self.conv_layers.append(tf.keras.layers.Conv2D(self.filt_out, kernel_size, padding='same'))
        self.norm_layers.append(tf.keras.layers.BatchNormalization())

        self.x_shortcut = []
        if self.conv_block:
            self.x_shortcut = [tf.keras.layers.Conv2D(self.filt_out, kernel_size,padding='same',strides=stride),
                               tf.keras.layers.BatchNormalization()]
        

        

    def call(self, input_tensor, training=False):
        x_in = input_tensor
        x = input_tensor

        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.norm_layers[i](x, training=training)
            x = tf.keras.layers.Activation('relu')(x)

        if self.conv_block:
            x_in = self.x_shortcut[0](input_tensor)
            x_in = self.x_shortcut[1](x_in,training=training)

        x = x+x_in
        

        return tf.keras.layers.Activation('relu')(x)
        
    def get_config(self):
        return {"kernel_size": self.kernel_size, "filt":self.filt,"filt_in":self.filt_in,"filt_out":self.filt_out,"num":self.num,"conv_block":self.conv_block,"stride":self.stride,"bottleNeck":self.bottleNeck}



def create_model(num_classes,transformation):
    model = tf.keras.Sequential([
        transformation,
        tf.keras.layers.Conv2D(32, 7,padding='same',strides=1),#56x56
        ResnetBlock(5,filt=64,filt_in=32,num=2,stride=2),#28x28
        ResnetBlock(5,filt=64,num=2),

        ResnetBlock(3,filt=128,filt_in=64,num=2,stride=2),#14x14
        ResnetBlock(3,filt=128,filt_in=128,num=2),

        ResnetBlock(3,filt=256,filt_in=128,num=2,stride=2),#7x7
        ResnetBlock(3,filt=256,filt_in=256,num=2),

        tf.keras.layers.AveragePooling2D(7),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes)
        ])

    model(tf.ones((1,*IMG_DIM)))
    return model



