import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from dataTransformer import *
import tensorflow as tf
from keras.preprocessing.image import image_dataset_from_directory
import os
from PIL import Image
from model_params import *
from dataTransformer import *
import time
import cv2







if __name__ == '__main__':
    real_images_paths = glob.glob("B/*")
    real_images = []
    for path in real_images_paths:
        imageName = path.split(os.path.sep)[-1]
        x,y,width,_ = imageName.split(" ")
        x = int(x)
        y = int(y)
        width = int(width)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        real_images.append(img[x:min(x+width,img.shape[0]),y:min(y+width,img.shape[1])])


    plt.figure(figsize=(10, 10))
    
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow( real_images[i])
        #plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()