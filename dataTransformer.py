import tensorflow as tf
from model_params import *
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import image_dataset_from_directory

class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filt, filt_in=None,filt_out=None,num=3,conv_block=False,stride=1,bottleNeck=True, name=None):
        super(ResnetBlock, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.filt = filt
        self.num = num
        self.bottleNeck=bottleNeck
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


def create_model(num_classes):
    model = tf.keras.Sequential([
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

    
    model.build([None,*INPUT_SHAPE])
    return model




@tf.function
def preprocess(image,label,train=False):
    
    #if train:
        #image = tf.image.random_crop(image,(len(image),CROP_SHAPE[0],CROP_SHAPE[0],3))
    image = tf.image.resize(image, (int(INPUT_SHAPE[0]), int(INPUT_SHAPE[1])))/255.0
    if train:
        
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        image = tf.image.random_hue(image,0.5)
        image = tf.image.random_saturation(image,0.5,1.5)
        image = tf.image.random_brightness(image,0.2)
        image = tf.image.random_contrast(image,0.5,1.5)
        
        
        
        
    return image,label

@tf.function
def load_resize_image(filename,label,img_dim=IMG_DIM):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, (int(img_dim[0]), int(img_dim[1])))
    
    return image,label

# Function to preprocess images
def resize_image(image, label,img_dim=IMG_DIM):
    image = tf.image.resize(image, (int(img_dim[0]), int(img_dim[1])))
    return image, label

# Function to load dataset from directory
def load_dataset(path:str,
                 train_val_split:tuple=TRAIN_VAL_SPLIT, 
                 batch_size:int=BATCH_SIZE, 
                 img_dim:tuple=IMG_DIM, 
                 seed:int=None,
                 cacheDir:str=CACHE_DIR,
                 parrallelCalls:int=tf.data.experimental.AUTOTUNE):
    
    
    assert(len(train_val_split)==2)
    assert(train_val_split[0]+train_val_split[1] < 1.0)

    train_ds = image_dataset_from_directory(path,shuffle=True,
                                           validation_split=2*train_val_split[1],
                                           subset='training',
                                           image_size=(img_dim[0], img_dim[1]),
                                           batch_size=None,
                                           seed=seed)
    
    dataset_val_test = image_dataset_from_directory(path,shuffle=True,
                                           validation_split=2*train_val_split[1],
                                           subset='validation',
                                           image_size=(img_dim[0], img_dim[1]),
                                           batch_size=batch_size,
                                           seed=seed)
    
    

    val_size = len(dataset_val_test)//2
    test_size = len(dataset_val_test) - val_size
    
    val_ds = dataset_val_test.take(val_size)
    test_ds = dataset_val_test.skip(val_size).take(test_size)
    
    
    if cacheDir != None:
        
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()
    
    
    val_ds = val_ds.prefetch(parrallelCalls)
    test_ds = test_ds.prefetch(parrallelCalls)
    return train_ds, val_ds, test_ds, dataset_val_test.class_names

def preprocess_ds(train_ds:tf.data.Dataset, preprocessing_function:callable, parrallelCalls:int=tf.data.experimental.AUTOTUNE, cacheFile:str=None,train=False):
    train_ds = train_ds.map(lambda image, label: (preprocessing_function(image,label,train)),num_parallel_calls=parrallelCalls)
    
    
    return train_ds
    
def evaluate_model(model:tf.keras.Model, test_ds:tf.data.Dataset, verbose:int=1, savePath:str=None):
    
    results = model.evaluate(test_ds)
    if verbose > 0:
        print("Test loss: "+str(results[0]))
        print("Test accuracy: "+str(results[1]))

    if savePath != None:
        model.save(savePath)
        if verbose > 0:
            print("Model saved to "+savePath)
        
    return results

def learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
def plot_learning_curve(history:tf.keras.callbacks.History, metric:str='accuracy', title:str='Learning Curve',save_fig:bool=False):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save_fig:
        plt.savefig("learning curves/"+title+'.png')

def confusion_matrix(model:tf.keras.Model, test_ds:tf.data.Dataset, normalize:bool=True, title:str='Confusion Matrix',save_fig:bool=False):
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = [label.numpy() for _, label in test_ds.unbatch()]
    y_true = tf.constant(y_true)
    
    print(y_true)
    cm = tf.math.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_fig:
        plt.savefig("confusion/"+title+'.png')
    
    