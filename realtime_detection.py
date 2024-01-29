import tensorflow as tf
import cv2
import argparse
import os
import time
import model_params


def record_camera(model, duration=10):
    cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

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

    while cv2.getTickCount() < end_time:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame from camera")
            break
        model_input = model_params.preProcessing(frame)
        model.predict(model_input)
        # ADD PRINT RESULT
        #

        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)

        

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Record screen or camera input.")
    parser.add_argument("mode", choices=["screen", "camera"], help="Mode: 'screen' or 'camera'")
    parser.add_argument("duration", type=int, help="Duration in seconds")
    parser.add_argument("--folder", help="Folder name to store the video")
    args = parser.parse_args()

    # Create the folder if it doesn't exist
    model = tf.keras.models.load_model('my_model.keras')
    record_camera(model)

    