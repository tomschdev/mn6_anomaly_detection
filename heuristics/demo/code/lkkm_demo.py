import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from random import shuffle
from sklearn.cluster import MiniBatchKMeans as mbKM
import math
import pandas as pd
from tqdm import tqdm

LKKM_PATH = 'heuristics/demo/score/lkkm'
VERSION = 'lkkm_score'

AVG = 50
CNT = 1
WDW = 16
PER = 10

kmeans = mbKM(n_clusters = 5, init = 'k-means++')

def calc_data_vectors(first, second, width, height):
    data_vecs = []
    for f, s in zip(first, second):
        vec = []
        vec.append(round(f[0]/width, 4))
        vec.append(round(f[1]/height, 4))
        vec.append(round(s[0]/width, 4))
        vec.append(round(s[1]/height, 4))
        data_vecs.append(np.asarray(vec))
    return data_vecs

def getInertiaElbow(col):
    col1 = col[0] - (col[1] - col[0])
    coln = col[-1:] + (col[-1:] - col[-2:-1])
    col = col.tolist()
    col.append(np.squeeze(coln))
    col.insert(0, col1)

    rng = np.arange(len(col)-2)
    deltas = []
    for i in range(1, len(col)-1):
        deltas.append((col[i+1]-col[i])-(col[i]-col[i-1]))
    return deltas, deltas.index(max(deltas))

# VERIFIED
def calc_ncd(recent, rnd, p_elbow):
    global kmeans
    preds = kmeans.predict(recent)
    
    #accumulate metrics per centroid config (elbow lead up)
    centroids = kmeans.cluster_centers_
    max_dist = []
    for x, c in zip(range(len(recent)), range(len(preds))):
        centr = centroids[preds[c]] 
        vec = recent[x]
        
        dist = np.linalg.norm(np.subtract(centr, vec))**2 
        if len(max_dist) < AVG:
            max_dist.append(dist)
        else:
            minmax = sorted(max_dist)[0]
            if dist > minmax:
                ind = max_dist.index(minmax)
                max_dist[ind] = dist
    
    if rnd % 5 == 0:
        record = []
        for i in range(2, 15):
            clmetrics = []
            kmeans.partial_fit(recent)
            preds = kmeans.labels_

            centroids = kmeans.cluster_centers_
            clmetrics.append(i)

            intra = kmeans.inertia_
            clmetrics.append(round(math.sqrt(intra),4))
            record.append(clmetrics)

        record = np.asarray(record)
        deltas, best_ind = getInertiaElbow(record[:, 1].copy())
        elbow = int(record[best_ind][0])
        kwargs = {
            'n_clusters': elbow,
        }
        kmeans.set_params(**kwargs)
    else:
        elbow = p_elbow
        
    #we predict before partial fit so as to keep recent data unseen
    kmeans.partial_fit(recent)
    return np.average(max_dist), elbow
    

def process_videos(videos):
    global kmeans
    print(videos)
    vid_scores = {}
    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    for vid in tqdm(videos):
        if vid.endswith(".mp4"):
            print(vid)
            cap = cv2.VideoCapture(vid)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            ret, first_frame = cap.read()
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

            count = 1
            elbow = None
            vector_store = []
            score_store = []
            rnd = 0

            while(cap.isOpened()):
                # cap.set(1, count)
                ret, frame = cap.read()
                # count += CNT
                if ret:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        next1, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
                        good_old = prev[status == 1]
                        good_new = next1[status == 1]

                        dv = calc_data_vectors(good_old, good_new, width, height)
                        vector_store.extend(dv)

                        if rnd > 1:
                            score, elbow = calc_ncd(dv, rnd, elbow)
                            score_store.append(score) 

                        else:
                            kmeans.partial_fit(dv)
                    except:
                        if len(score_store) != 0:
                            score_store.append(max(score_store))
                        else:
                            score_store.append(0)

                    rnd+=1
                    prev_gray = gray.copy()
                    prev = good_new.reshape(-1, 1, 2)
                    
                else:
                    kmeans = None
                    kmeans = mbKM(n_clusters = 5, init = 'k-means++', random_state = 42)
                    break
            cap.release()
            
            avgd_data  = []
            wdw = WDW
            sz  = len(score_store)/wdw
            for i in range(1, int(sz)):
                if i == sz-1:
                    avgd_data.append(np.average(score_store[(i-1)*wdw:]))
                else:
                    avgd_data.append(np.average(score_store[(i-1)*wdw: i*wdw]))
                    
            pname = os.path.split(vid)[1]
            pname = pname.split('.')[0]
            vid_scores["{}".format(pname)] = avgd_data
            
    return vid_scores

def main():
    videos = ["turn.mp4"]
    videos = [os.path.join("heuristics/demo/sample", v) for v in videos]
    
    scores = process_videos(videos)
    score_dict = json.dumps(scores)
    
    print("[SPARSE_FLOW] predictions produced from loaded model")
    with open(os.path.join(LKKM_PATH, str(VERSION + ".json")), "w") as jf:
        print("[TEST_NET] sim written to {}".format(LKKM_PATH))
        json.dump(score_dict, jf)

    fig, ax = plt.subplots(1,len(scores), squeeze=False)
    cnt = 0
    for v, s in scores.items():
        ax[0, cnt].plot(np.arange(len(s)),s)
        ax[0, cnt].set_xlabel("LKKM SCORE PROFILE: " + v)
        cnt+=1
    fig.savefig(os.path.join(LKKM_PATH, str(VERSION + ".png")))
    plt.show()

if __name__ == '__main__':
    main()
