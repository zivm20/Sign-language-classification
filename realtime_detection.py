import tensorflow as tf
import cv2
import argparse
import os
import time
from model_params import *
import numpy as np
from dataTransformer import *


def create_transformation_layer():
    transforms = []
    transforms.append(tf.keras.layers.Rescaling(1./255))
    transforms.append(tf.keras.layers.CenterCrop(IMG_DIM[0]-10,IMG_DIM[1]-10))
    transforms.append(tf.keras.layers.RandomRotation(ROTATION_FACTOR))
    transforms.append(tf.keras.layers.RandomCrop(CROP_SHAPE[0],CROP_SHAPE[1]))
    transforms.append(tf.keras.layers.Resizing(INPUT_SHAPE[0],INPUT_SHAPE[1]))
    transforms.append(tf.keras.layers.RandomContrast(CONTRAST_FACTOR))
    
    return transforms 

def record_camera(model, duration=10):
    cap = cv2.VideoCapture(1)  # Change the index if you have multiple cameras

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Failed to open camera")
        return

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20.0
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    start_time = cv2.getTickCount()
    end_time = start_time + duration * cv2.getTickFrequency()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame from camera")
            break
        bbox = (300,300,128,128)
        model_input = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        #model_input = preProcessing(model_input)
        pred = model(preProcessing(model_input))
        score = tf.nn.softmax(pred[0])
        className = CLASS_NAMES[np.argmax(score)]
        
        frame = cv2.putText(frame,className,org=(int(bbox[0]+bbox[3]/2),bbox[1]+bbox[2]),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,thickness=2,color=(255,0,0))
        frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(255,0,0),2)
        
        

        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)
        # Display the live camera feed
        cv2.imshow("model", model_input)
        key = 0xFF & cv2.waitKey(1)
        if key == ord('q'):
            break

        

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse command-line arguments
    print(tf.config.list_physical_devices('GPU'))

    # Create the folder if it doesn't exist
    #model = create_model(len(CLASS_NAMES),create_transformation_layer())
    model = create_model(len(CLASS_NAMES))
    model.load_weights('resnet_model.keras')
    record_camera(model)

    