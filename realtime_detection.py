import tensorflow as tf
import cv2
import argparse
import os
import time
from model_params import *
import numpy as np
from dataTransformer import *

bboxY = 200
bboxX = 200



def record_camera(models, duration=10):
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
        bbox = (bboxY,bboxX,128,128)
        model_input = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        #model_input = preProcessing(model_input)
        colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
        for i, model in enumerate(models):
            pred = model(preProcessing(model_input))
            score = tf.nn.softmax(pred[0])
            className = CLASS_NAMES[np.argmax(score)]
            frame = cv2.putText(frame,className,org=(int(bbox[0]+bbox[3]/2),bbox[1]+bbox[2]+i*50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,thickness=2,color=colors[i%len(colors)])
        
        
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
    resnet_model = create_resnet_model(len(CLASS_NAMES))
    resnet_model.load_weights(MODEL_DIR+'/resnet_model.keras')
    simple_cnn_model = create_simple_cnn_model(len(CLASS_NAMES))
    simple_cnn_model.load_weights(MODEL_DIR+'/simple_cnn_model.keras')
    nn_model = create_nn_model(len(CLASS_NAMES))
    nn_model.load_weights(MODEL_DIR+'/nn_model.keras')
    lr_model = create_lr_model(len(CLASS_NAMES))
    lr_model.load_weights(MODEL_DIR+'/lr_model.keras')

    record_camera([resnet_model,simple_cnn_model,nn_model,lr_model])

    