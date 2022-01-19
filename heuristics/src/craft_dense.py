import json
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from tf_raft.model import RAFT, SmallRAFT
from tf_raft.losses import sequence_loss, end_point_error
from tf_raft.losses import sequence_loss, end_point_error
from tf_raft.datasets import KITTI, ShapeSetter, CropOrPadder
from tf_raft.datasets import dataset as data
from tf_raft.datasets import flow_viz as fv
import matplotlib.pyplot as plt
import cv2
import tensorflow_addons as tfa
import os
from tf_raft.training import VisFlowCallback, first_cycle_scaler
from random import shuffle
from sklearn.preprocessing import StandardScaler

ITERS = 24
ITERS_P = 12

sim_path = 'results/sim'
delta_path = 'results/delta'
version = 'kraken_normv2_ms'


def calculateDistance(pr2, x2vis, x2, x1):
    delta = (x2-x1)**2
    pred_err = (x2vis-pr2)**2
    pred_err_dist = np.squeeze(np.sum(pred_err))
    delta_dist = np.squeeze(np.sum(delta))
    
    return pred_err_dist, delta_dist #prediction error per unit change NB
   

def process_videos(model, videos):
    
    vid_sim = {}
    vid_delta = {}
    dim = (112, 112)
    
    for path in tqdm(videos):
        print("processing: {}".format(path))
        count = 0
        vid = cv2.VideoCapture(path)
        if vid.isOpened()== False:
            print("error opening video")
        ret, x1 = vid.read()
        x1 = cv2.resize(x1, dsize=dim, interpolation=cv2.INTER_LINEAR)
        
        hsv = np.zeros_like(x1)
        hsv[...,1] = 255
        sim = []
        sim_buff = []
        delta = []
        delta_buff = []
        # plt.ion()
        # plt.show()
        while vid.isOpened():
            vid.set(1, count)
            ret, x1 = vid.read()
            ret, x2 = vid.read()
            count += 8
            
            if ret:
                x1 = cv2.resize(x1, dsize=dim, interpolation=cv2.INTER_LINEAR)
                x2 = cv2.resize(x2, dsize=dim, interpolation=cv2.INTER_LINEAR)
                #x1 = cv2.GaussianBlur(x1, (5,5), 0)
                #x2 = cv2.GaussianBlur(x2, (5,5), 0)
                # cv2.imshow('Frame',x1)
                # cv2.imshow('Frame',x2)
                # x2 = np.expand_dims(x2, axis=0)
                # x1 = np.expand_dims(x1, axis=0)
                # print(x1.shape)
                # print(x2.shape)
                
                flow = model([np.expand_dims(np.asarray(x1, dtype=np.float32), axis=0), np.expand_dims(np.asarray(x2, dtype=np.float32), axis=0)], training=False)
                flow = np.squeeze(np.squeeze(np.asarray(flow, dtype=np.float32)))
                
                # FARNEBACK EXPERIMENT
                # x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2GRAY)
                # x2 = cv2.cvtColor(x2, cv2.COLOR_RGB2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(x1, x2, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
                
                h = flow.shape[0]
                w = flow.shape[1]
                flow[:,:,0] += np.arange(w)
                flow[:,:,1] += np.arange(h)[:,np.newaxis]
                pred_frame = cv2.remap(x1, flow, None, cv2.INTER_LINEAR)
                pred_frame = cv2.GaussianBlur(pred_frame, (5,5), 0) 
                x2_vis = cv2.GaussianBlur(x2, (5,5), 0)
                
                # cv2.imshow("pred", pred_frame)
                # print(pred_frame.shape)
                # diff = pred_frame - x2_vis
                # plt.imshow(diff)
                # plt.title("pred x2")
                # plt.draw()
                # plt.pause(0.00001)
                # plt.show(block=False)
                # plt.imshow(x2_vis)
                # plt.title("x2")
                # plt.draw()
                # plt.pause(0.00001)
                # plt.show(block=False)

                pred_err, diff = calculateDistance(pred_frame, x2_vis, x2, x1)
                
                if count % 16 == 0:
                    sim_buff.append(pred_err)
                    delta_buff.append(diff)
                    sim.append(int(np.average(sim_buff)))
                    delta.append(int(np.average(delta_buff)))
                    sim_buff = []
                    delta_buff = []
                else:
                    sim_buff.append(pred_err)                
                    delta_buff.append(diff)
                
            else:
                vid.release()
                break
            
       
        pname = os.path.split(path)[1]
        pname = pname[:-9]
        pcat = "Normal"
        pnum = pname[-3:]

        vid_sim["{}/{}".format(pcat, pnum)] = sim
        vid_delta["{}/{}".format(pcat, pnum)] = delta
        print("{}/{} completed".format(pcat, pnum))
    return vid_sim, vid_delta

def main():
    raftms = RAFT(iters=ITERS, iters_pred=ITERS_P)
    raftms.load_weights("model/mpi_sintel/model").expect_partial()
    #raftfc = RAFT(iters=ITERS, iters_pred=ITERS_P)
    #raftfc.load_weights("model/flying_chairs/model").expect_partial()
    #raftki = RAFT(iters=ITERS, iters_pred=ITERS_P)
    #raftki.load_weights("model/kitti10/model").expect_partial()

    videos = os.listdir("normal-videos-testing")
    videos = [os.path.join("normal-videos-testing", v) for v in videos]
    sim, delta  = process_videos(raftms, videos)
    raftfc_sim = sim
    sim_dict = json.dumps(sim)
    delta_dict = json.dumps(delta)
    print("[TEST_NET] predictions produced from loaded model")
    print(sim_dict)
    with open(os.path.join(sim_path, version + "_sim.json"), "w") as jf:
        print("[TEST_NET] sim written to {}".format(sim_path))
        json.dump(sim_dict, jf)
    
    with open(os.path.join(delta_path, version + "_delta.json"), "w") as jf:
        print("[TEST_NET] delta written to {}".format(delta_path))
        json.dump(delta_dict, jf)

    
    fig, ax = plt.subplots(1,len(raftfc_sim), squeeze=False)
    cnt = 0
    for v, s in raftfc_sim.items():
        ax[0, cnt].plot(np.arange(len(s)),s)
        ax[0, cnt].set_xlabel("sim: " + v)
        cnt+=1
        
    plt.show()
if __name__ == '__main__':
    main()
