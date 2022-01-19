import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import tensorflow_addons as tfa
import os
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans as mbKM
import math
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from tqdm import tqdm

lkkm_path = 'results/lkkm'
version = 'kraken_lkkm_anom_consec_v2.json'

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
     
    # accumulate metrics per centroid config (elbow lead up)
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
            
            # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            
            prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

            # mask = np.zeros_like(first_frame, dtype=np.uint8)

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


                        # Draws the optical flow tracks
                        # for i, (new, old) in enumerate(zip(good_new, good_old)):
                            # Returns a contiguous flattened array as (x, y) coordinates for new point
                            # a, b = new.ravel()
                            # Returns a contiguous flattened array as (x, y) coordinates for old point
                            # c, d = old.ravel()
                            # Draws line between new and old position with green color and 2 thickness
                            # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                            # frame = cv2.circle(frame, (int(a), int(b)), 3, color, -1)
                        # Overlays the optical flow tracks on the original frame
                        # output = cv2.add(frame, mask)
                        # output = mask
                        # cv2.imshow("sparse optical flow", output)

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
            pname = pname[:-9]
            pcat = pname[:-3]
            # pcat = "Normal"
            pnum = pname[-3:]       
            print("{}/{}".format(pcat, pnum))
            vid_scores["{}/{}".format(pcat, pnum)] = avgd_data
               
    return vid_scores

def main():
    
    videos = os.listdir("test-video-set")
    videos = [os.path.join("test-video-set", v) for v in videos]

    scores = process_videos(videos)
    
    score_dict = json.dumps(scores)
    print("[SPARSE_FLOW] predictions produced from loaded model")
    with open(os.path.join(lkkm_path, version), "w") as jf:
        print("[TEST_NET] sim written to {}".format(lkkm_path))
        json.dump(score_dict, jf)


if __name__ == '__main__':
    main()
