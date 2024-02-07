import tensorflow as tf
import cv2
import argparse
import os
import time
from model_params import *
import numpy as np

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
        model_input = preProcessing(frame)
        pred = model.predict(model_input)
        score = tf.nn.softmax(pred[0])
        className = CLASS_NAMES[np.argmax(score)]
        org = (200,200)
        frame = cv2.putText(frame,className,org = org,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,thickness=2,color=(255,0,0))
        

        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)
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
    model = tf.keras.models.load_model('my_model.keras')
    
    record_camera(model)

    