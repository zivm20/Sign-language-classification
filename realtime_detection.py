import tensorflow as tf
import cv2
import argparse
import os
import time
from model_params import *
import numpy as np
from dataTransformer import *
from models import *
import glob

bboxY = 200
bboxX = 200



def record_camera(models:list[tf.keras.Model], duration=10,correct_duration = 3,src=None, saveDir = 'output',bboxY = 200, bboxX = 200, save_correct = True):
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
    
    class_paths = []
    displayImg = False
    sample = None
    sample_class = None
    class_real =None
    sample_size = (128,128)

    if src is not None and duration > 0:
        class_paths = [src + f for f in os.listdir(src)]
        displayImg = True
        sample_class = np.random.choice(class_paths)
        sample = cv2.imread(np.random.choice(glob.glob(sample_class+'/*.jpg')))
        sample = cv2.resize(sample,sample_size)
        class_real = sample_class.split('/')[-1]

    correct_color = (0,255,0)
    wrong_color = (0,0,255)
    
    saveNum = 0
    model_names= ['resnet','cnn','nn','lr'] 
    correct_total = 0
    model_correct_num = {name:0 for name in model_names}
    correct_once = False
    correct_start = cv2.getTickCount()
    scale_x = 2
    scale_y = 2
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break

        
            
        correct = False
        
        bbox = (bboxY,bboxX,128,128)
        model_input = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        
        pred_classes = {}
        pred_scores = {}
        for i, model in enumerate(models):
            pred = model(preProcessing(model_input),training=False)
            score = tf.nn.softmax(pred[0])
            
            pred_scores[model_names[i]] = np.max(score)
            className = CLASS_NAMES[np.argmax(score)]
            pred_classes[model_names[i]] = className
            if className == class_real:
                correct = True
                correct_once = True
        
        color = correct_color
        if not correct:
            color = wrong_color
            

        if sample is not None:
            frame[0:sample_size[1],0:sample_size[0]] = sample
        frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color,2)
        old_frame_dim = (frame.shape[1],frame.shape[0])
        frame = cv2.resize(frame,(int(frame.shape[1]*scale_x),int(frame.shape[0]*scale_y)))
                           
        if correct_once:
            correct_total+=1
            for model_name in model_names:
                if pred_classes[model_name] == class_real:
                    model_correct_num[model_name]+=1
        model_accuracy = {name:0 for name in model_names}
        
        for i,model_name in enumerate(model_names):
            color = wrong_color
            if pred_classes[model_name] == class_real:
                color = correct_color
            accuracy = 0
            if correct_total > 0:
                accuracy = model_correct_num[model_name]/correct_total
            model_accuracy[model_name] = accuracy
            if sample is not None:
                frame = cv2.putText(frame, f"{model_name} accuracy: {accuracy:.2%}",org=(scale_x*(sample_size[0]+10),scale_y*(i+1)*25),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5*max(scale_x,scale_y),thickness=1*max(scale_x,scale_y),color=color)
            frame = cv2.putText(frame,f"{model_name}: {CLASS_NAMES.index(pred_classes[model_name])}:{pred_classes[model_name]}",org=(scale_x*(int(bbox[0]+bbox[3])+10),scale_y*(bbox[1]+(i+1)*25)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5*max(scale_x,scale_y),thickness=1*max(scale_x,scale_y),color=color)
            frame = cv2.putText(frame,f"{model_name} certainty: {pred_scores[model_name]:.2%}",org=(scale_x*(25),scale_y*(sample_size[1]+(i+2)*25)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5*max(scale_x,scale_y),thickness=1*max(scale_x,scale_y),color=color)
        color = correct_color
        if not correct:
            color = wrong_color
        if sample is not None:
            frame = cv2.putText(frame,f"{CLASS_NAMES.index(class_real)}:{class_real}",org=( scale_x*(int(sample_size[0]/2)-25),scale_y*(sample_size[1]+25)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1*max(scale_x,scale_y),thickness=1*max(scale_x,scale_y),color=color)
        
        if not correct_once:
            correct_start = cv2.getTickCount()
        
        # Display the live camera feed
        cv2.imshow("Camera Feed", frame)
        # what the model sees
        cv2.imshow("model", model_input)
        key = 0xFF & cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s') or (cv2.getTickCount() > correct_start + correct_duration * cv2.getTickFrequency() and displayImg and save_correct and ( max(model_accuracy.values()) > 0.9 or sum(model_accuracy.values())>1.7)):
            cv2.imwrite(f'{saveDir}/frame_{saveNum}.jpg',cv2.resize(frame,old_frame_dim))
            saveNum+=1

        if ((cv2.getTickCount() > end_time) or cv2.getTickCount() > correct_start + correct_duration * cv2.getTickFrequency())and displayImg:
            start_time = cv2.getTickCount()
            end_time = start_time + duration * cv2.getTickFrequency()
            if sample is not None:
                sample_class = np.random.choice(class_paths)
                sample = cv2.imread(np.random.choice(glob.glob(sample_class+'/*.jpg')))
                sample = cv2.resize(sample,sample_size)
                class_real = sample_class.split('/')[-1]
                correct_total = 0
                model_correct_num = {name:0 for name in model_names}
                correct_once = False
            

        

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

    record_camera([resnet_model,simple_cnn_model,nn_model,lr_model],src="asl_alphabet_train/")

    