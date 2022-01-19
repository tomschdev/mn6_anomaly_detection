import json
import numpy as np
from tqdm import tqdm
from tf_raft.model import RAFT
import matplotlib.pyplot as plt
import cv2
import os


SIM_PATH = 'heuristics/demo/score/craft'
VERSION = 'craft_score'

def calculateDistance(pr2, x2vis, x2, x1):
    pred_err = (x2vis-pr2)**2
    pred_err_dist = np.squeeze(np.sum(pred_err))
    return pred_err_dist

def process_videos(model, videos):
    
    vid_sim = {}
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
        
        while vid.isOpened():
            # vid.set(1, count)
            ret, x1 = vid.read()
            ret, x2 = vid.read()
            # count += 1
            if ret:
                x1 = cv2.resize(x1, dsize=dim, interpolation=cv2.INTER_LINEAR)
                x2 = cv2.resize(x2, dsize=dim, interpolation=cv2.INTER_LINEAR)
                flow = model([np.expand_dims(np.asarray(x1, dtype=np.float32), axis=0), np.expand_dims(np.asarray(x2, dtype=np.float32), axis=0)], training=False)
                flow = np.squeeze(np.squeeze(np.asarray(flow, dtype=np.float32)))
                h = flow.shape[0]
                w = flow.shape[1]
                flow[:,:,0] += np.arange(w)
                flow[:,:,1] += np.arange(h)[:,np.newaxis]
                pred_frame = cv2.remap(x1, flow, None, cv2.INTER_LINEAR)
                pred_frame = cv2.GaussianBlur(pred_frame, (5,5), 0) 
                x2_vis = cv2.GaussianBlur(x2, (7,7), 0)
                pred_err = calculateDistance(pred_frame, x2_vis, x2, x1)
                sim.append(int(pred_err))
                
                # if count % 16 == 0:
                #     sim_buff.append(pred_err)
                #     delta_buff.append(diff)
                #     sim.append(int(np.average(sim_buff)))
                #     delta.append(int(np.average(delta_buff)))
                #     sim_buff = []
                #     delta_buff = []
                # else:
                #     sim_buff.append(pred_err)                
                #     delta_buff.append(diff)
                
            else:
                vid.release()
                break
            
        pname = os.path.split(path)[1]
        pname = pname.split(".")[0]

        vid_sim["{}".format(pname)] = sim
        print("<{}> completed".format(pname))
    return vid_sim

def main():
    raftms = RAFT(iters=100, iters_pred=1)
    raftms.load_weights("heuristics/model/mpi_sintel/model").expect_partial()

    videos = ["train.mp4"]
    videos = [os.path.join("heuristics/demo/sample", v) for v in videos]
    sim = process_videos(raftms, videos)
    sim_dict = json.dumps(sim)
    
    print("[TEST_NET] predictions produced from loaded model")
    with open(os.path.join(SIM_PATH, str(VERSION + ".json")), "w") as jf:
        print("[TEST_NET] CRAFT sim written to {}".format(os.path.join(SIM_PATH, str(VERSION + ".json"))))
        json.dump(sim_dict, jf)
        
    
    fig, ax = plt.subplots(1,len(sim), squeeze=False)
    cnt = 0
    for v, s in sim.items():
        ax[0, cnt].plot(np.arange(len(s)),s)
        ax[0, cnt].set_xlabel("CRAFT SCORE PROFILE: " + v)
        cnt+=1
    fig.savefig(os.path.join(SIM_PATH, str(VERSION + ".png")))
    plt.show()
    
if __name__ == '__main__':
    main()
