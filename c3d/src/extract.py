from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import skvideo.io
from c3d import C3D
from keras.models import Model
from sklearn import preprocessing
import numpy as np
import json
import subprocess
import os
import cv2
from tqdm import tqdm
from scipy.misc import imresize
import tensorflow as tf
import warnings
import json
import time
import warnings
warnings.filterwarnings('ignore')

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    frames = np.asarray(frames)
    print(frames.shape)
    return frames

def preprocess_input(video):
    intervals = np.ceil(np.linspace(0, video.shape[0]-1, 16)).astype(int)
    frames = video[intervals]
    
    reshape_frames = np.zeros((frames.shape[0], 112, 112, frames.shape[3]))
    for i, img in enumerate(frames):
        img = imresize(img, (112,112), 'bicubic')
        reshape_frames[i,:,:,:] = img
        
    reshape_frames = np.expand_dims(reshape_frames, axis=0)
    
    return reshape_frames

# VERIFIED
def extract_batch(dest_folder, path_to_video_set, model):
    print("STARTING EXTRACTION")
    print("VIDEOS FROM: {}".format(path_to_video_set))
    print("FEATURES TO: {}".format(dest_folder))

    f = open("annMIL/misc/training_videos.txt", "r")
    lines = f.readlines()
    train = [x.strip() for x in lines]

    with tf.device("/GPU:0"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            video_list = os.listdir(path_to_video_set)
            start = time.time()
            for video in tqdm(video_list):
                print(video)
                video_features_stack = {}
                short = video[:-9]
                if short.startswith("Normal"):
                    cat = "Normal"
                else:
                    cat = str(short[:-3]) 
                num = short[-3:]
                # filename = "{}{}.txt".format(cat,num)
                filename = video.split(".")[0]
                lbl = "{}/{}.txt".format(cat,num)
                print(filename)
                if lbl not in train:
                    vid = get_video_frames(os.path.join(path_to_video_set, video))

                    #extract all features
                    video_features = []
                    count = 0
                    for i in range(16, len(vid), 16):
                        count += 1
                        x = preprocess_input(vid[i-16:i]) 
                        features = model.predict(x)
                        assert (features.shape == (1,4096))
                        norm_features = preprocessing.normalize(features, norm='l2')
                        video_features.append(norm_features)
                    video_features = np.asarray(video_features).reshape(len(video_features), 4096)
                    print("[EXTRACT_NET] features extracted: shape {}".format(video_features.shape))

                    # calculate average vector for 32 segments
                    split_sgms = np.array_split(video_features, 32)
                    avgd_split_sgms = []
                    for features in split_sgms:
                        avgd_split_sgms.append(np.round_(np.mean(features, axis=0, dtype=np.float32), decimals=4))

                    print("[EXTRACT_NET] features averaged into 32: shape ({}, {})".format(len(avgd_split_sgms), len(avgd_split_sgms[0])))

                    video_features_stack = np.asarray(avgd_split_sgms).reshape(32, 4096)
                    video_features_stack = np.nan_to_num(video_features_stack)
                    np.savetxt(os.path.join(dest_folder, str(filename + ".txt")), video_features_stack, fmt='%.4f', delimiter=" ")
                    print("[EXTRACT_NET] features written to {}".format(os.path.join(dest_folder, filename)))
                
                else:
                    print("[EXTRACT NET] skipping, not in test set")
            end = time.time()
            # f = open(os.path.join("time", dest_folder.split("/")[1]), "w")
            # f.write("feature extraction of {} videos on GPU took {} sec".format(len(video_list), (end-start)))
            # f.close()

def extract(to_features, from_videos):
    base_model = C3D()
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)
    extract_batch(to_features, from_videos, model)

if __name__ == '__main__':
    extract("c3d/feature", "c3d/sample")
    
    
    
    


        
    
