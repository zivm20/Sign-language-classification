import tensorflow as tf
from model_params import *
import numpy as np
import matplotlib.pyplot as plt

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


def create_resnet_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 7,padding='same',strides=2),#56x56
        ResnetBlock(5,filt=64,filt_in=32,num=2,stride=2),#28x28
        ResnetBlock(5,filt=64,filt_in=64,num=2),

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

def create_simple_cnn_model(num_classes):
    simple_cnn_model = tf.keras.Sequential([
  
        tf.keras.layers.Conv2D(32, 7,padding='same',strides=2),#56x56
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),#28x28

        tf.keras.layers.Conv2D(64, 5,padding='same',strides=1),#28x28
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),#14x14

        tf.keras.layers.Conv2D(128, 3,padding='same',strides=1),#14x14
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),#7x7
        
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes)
    ])
    simple_cnn_model.build([None,*INPUT_SHAPE])
    return simple_cnn_model

def create_nn_model(num_classes):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Resizing(INPUT_SHAPE[0]//2,INPUT_SHAPE[1]//2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Normalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Normalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Normalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    nn_model.build([None,*INPUT_SHAPE])
    return nn_model

def create_lr_model(num_classes):
    lr_model = tf.keras.Sequential([
        tf.keras.layers.Resizing(INPUT_SHAPE[0]//2,INPUT_SHAPE[1]//2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes)
    ])
    lr_model.build([None,INPUT_SHAPE[0]//2,INPUT_SHAPE[1]//2,INPUT_SHAPE[2]])
    return lr_model



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


    
def plot_learning_curve(history:tf.keras.callbacks.History,subplot:plt.Axes=None, metric:str='accuracy', title:str='Learning Curve',save_fig:bool=False):
    if subplot != None:
        subplot.plot(history.history[metric])
        subplot.plot(history.history['val_'+metric])
        subplot.set_title(title)
        subplot.set_xlabel(metric)
        subplot.set_xlabel('epoch')
        subplot.legend(['train', 'val'], loc='upper left')
    if subplot == None:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+metric])
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save_fig:
            plt.savefig("learning curves/"+title+'.png')

def plot_curves(history:tf.keras.callbacks.History,save_fig:bool=False,title:str='Model Learning Curves'):
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    plt.suptitle(title)
    plot_learning_curve(history,subplot=axs[0],metric='accuracy',title='Accuracy')
    plot_learning_curve(history,subplot=axs[1],metric='loss',title='Loss')
    
    if save_fig:
        plt.savefig("learning curves/"+title+".png")
        
    

def confusion_matrix(model:tf.keras.Model, test_ds:tf.data.Dataset, normalize:bool=True, title:str='Confusion Matrix',save_fig:bool=False):
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = [label.numpy() for _, label in test_ds.unbatch()]
    y_true = tf.constant(y_true)
    
    print(y_true)
    cm = tf.math.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap='cool')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_fig:
        plt.savefig("confusion/"+title+'.png')