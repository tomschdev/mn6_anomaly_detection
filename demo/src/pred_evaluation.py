import time
import numpy as np
import os
import sys
from functools import reduce
from random import shuffle
import copy
import time
from itertools import islice
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import pandas as pd
import streamlit as st
import base64
from itertools import islice
from statsmodels.tsa.seasonal import seasonal_decompose
from roccer import compute_score_seq, compute_ROC
import altair as alt
from math import ceil
from PIL import Image


root=os.getcwd()
VIDEO_PATH="/home/tomsch/Honours/Honours_Project/test-video-set/all-videos" 
ANNO_FILE=os.path.join(root, "demo", "annotations", "Temporal_Anomaly_Annotation.txt")
min_thresh = 0.1
max_thresh = 0.9
CLASSES = ["A", "N"]
PER=10


def read_anno_file(f):
    anno = np.genfromtxt(f, dtype=str, comments="{% comment %}")
    return anno

def verify_anom_fc6(scores, n_frame, anom_window, threshold, prev_res):
    n_scores = 32
    start_anom = int(anom_window[0])
    end_anom = int(anom_window[1])
    start_anom = max(0, int((start_anom/n_frame)*n_scores - 4))
    end_anom = min(n_scores, ceil((end_anom/n_frame)*n_scores + 4))
    anno_sgms = sorted(list(set([max(0,start_anom-1), end_anom-1])))
    
    if len(anno_sgms) == 1:
        anno_sgms.append(anno_sgms[0]+1)

    for i in range(anno_sgms[0], anno_sgms[1]):
        if scores[i] >= threshold:
            
            return True, anno_sgms
    return (False or prev_res), anno_sgms
        

def frame_averaging(dataset):
    sims_d_div16 = {}
    for lbl, data in dataset.items():
        avgd_data  = []
        sz  = len(data)/16
        for i in range(1, int(sz)):
            if i == sz-1:
                avgd_data.append(np.average(data[(i-1)*16:]))
            else:
                avgd_data.append(np.average(data[(i-1)*16: i*16]))
        sims_d_div16[lbl] = avgd_data
    return sims_d_div16      

def individual_scale(values_d):
    scaled_values = {}
    for lbl, val in values_d.items():
        s = np.std(val)
        u = np.average(val)
        val_sc = [(x-u)/s for x in val]
        scaled_values[lbl] = val_sc
    return scaled_values
      
def filter_to_zero(values_d, t):
    filtered_values = {}
    for lbl, val in values_d.items():
        val_filt = list(map(lambda x: 0 if x < t else x, val))
        filtered_values[lbl] = val_filt
    return filtered_values

def relative_scale(sims, group, cutoff):
    scaled_sims = {}
    batch_values = []
    count = 0
    lbls = list(sims.keys())
    shuffle(lbls)
    
    for lbl in lbls:
        sim = sims[lbl]
        batch_values.extend(sim)
        count +=1
        if count % group == 0 or count == len(sims):
            if count % group == 0:
                subset = islice(sims.items(), count-group, count)
            else:
                back_count = 0
                while (int(count-back_count) % group != 0):
                    back_count+=1
                subset = islice(sims.items(), count-back_count, count)
                
                
            s = np.std(batch_values)
            u = np.average(batch_values)
            m = np.max(batch_values)
            ms = (m-u)/s
            
            for lb, si in subset:
                sim_sc = [(x-u)/s for x in si]
                sim_sc = np.clip(np.asarray(sim_sc), cutoff, np.max(sim_sc)) #atleast clip anom scores that are less than 1 std devs from avg of batch
                scaled_sims[lb] = sim_sc
            batch_values = []
            
    return scaled_sims
            
def to_range(values_d, r):
    range_d = {}
    for lbl, values in values_d.items():
        mx = float(max(values))
        mn = float(min(values))
        if int(mx) == 0:
            values_norm = [0 for x in values]
        else:
            values_norm = [float(r*(x-mn)/(mx-mn)) for x in values]
        range_d[lbl] = values_norm
    return range_d

def bulkrangeto1(sims, group, cutoff):
    scaled_sims = {}
    batch_values = []
    count = 0
    lbls = list(sims.keys())
    shuffle(lbls)
    
    for lbl in lbls:
        sim = sims[lbl]
        batch_values.extend(sim)
        count +=1
        if count % group == 0 or count == len(sims):
            if count % group == 0:
                subset = islice(sims.items(), count-group, count)
            else:
                back_count = 0
                while (int(count-back_count) % group != 0):
                    back_count+=1
                subset = islice(sims.items(), count-back_count, count)
                
            mx = np.max(batch_values)
            mn = np.min(batch_values)
            for lb, si in subset:
                sim_sc = [(x-mn)/(mx-mn) for x in si]
                scaled_sims[lb] = sim_sc
                
            batch_values = []
            
    return scaled_sims
    
def peak_window(scores_d, wdw):
    transform_scores_d = {}
    for lbl, scores in scores_d.items():
        transform_scores = []
        for f in range(0, len(scores)):
            if f > wdw and (len(scores)-f-1) > wdw:

                before = sum(scores[f-wdw:f])
                after = sum(scores[f:f+wdw])
                step = after - before
                # print("\t[PEAK WDW] valid f -> score is ", scores[f])
                # print("\t[PEAK WDW] valid f -> step is ", step)
                # print("\t[PEAK WDW] valid f -> transformed scores is ", scores[f]+step)
                transform_scores.append(max(scores[f], scores[f] + step))
            else:
                # print("\t[PEAK WDW] f not in valid wdw range")
                transform_scores.append(scores[f])
                
        transform_scores_d[lbl] = transform_scores
    return transform_scores_d

def delta_multiplier(sims_d, delta_d):
    mul_scores_d = {}
    for lbl, scores in sims_d.items():
        mul_scores = []
        deltas = delta_d[lbl]
        d_avg = np.average(deltas)
        for x, m in zip(scores, deltas):
            mul_scores.append((m/d_avg)*x)
        mul_scores_d[lbl] = mul_scores
    return mul_scores_d
    
def center_on_avg_err(sims_d):
    mul_scores_d = {}
    for lbl, scores in sims_d.items():
        mul_scores = []
        s_avg = np.average(scores)
        for x in scores:
            mul_scores.append(x-s_avg)
        mul_scores_d[lbl] = mul_scores
    return mul_scores_d

def convert_to_32(profile):
    profile_32 = {}
    for lbl, data in profile.items():
        new_data = []
        data = np.asarray(data)
        sgms = np.array_split(data, 32)
        for sgm in sgms:
            if len(sgm.tolist()) == 0:
                new_data.append(0)
            else:
                new_data.append(np.max(sgm))
        profile_32[lbl] = new_data
    return profile_32

def decompose(lkkm_d):
    dcmp = {}
    for lbl, lkkm in lkkm_d.items():
        if len(lkkm) > 10:
            score_frame = pd.DataFrame({
                "data": lkkm   
            })
            score_frame.fillna(0, inplace=True)
            result = seasonal_decompose(score_frame, model='additive', period=min(int(len(score_frame["data"])/2), PER))
            resid_frame = pd.DataFrame({
                "data": result.resid   
            })
            resid_frame.fillna(0, inplace=True)
            
            dcmp[lbl] = np.asarray(resid_frame["data"], dtype=np.float32)
        
    return dcmp

def nan_to_zero(values_d):
    filled_d = {}
    for lbl, val in values_d.items():
        df = pd.DataFrame({   
            "data": val,
        })
        df["data"] = df["data"].fillna(0)
        filled_d[lbl] = df["data"].values
    return filled_d

def broadc(vals, splt):
    broad_list = []
    sgms = np.array_split(vals, splt)
    for sgm in sgms:
        l = sgm.tolist()
        m = max(l)
        broad = [m]*int(32/splt)
        broad_list.extend(broad)
    assert len(broad_list) == 32
    return broad_list

def apply_broadc(values_d, splt):
    broad_d = {}
    for lbl, val in values_d.items():
        broad_d[lbl] = broadc(val, splt)
    return broad_d

def add_scores(vals1, vals2):
    combined = [x+y for x,y in zip(vals1, vals2)]
    return combined

def combine_heuristics(sims_d, lkkm_d):
    hrts_d = {}
    for lbl, lkkm in lkkm_d.items():
        craft = sims_d[lbl]
        lk = broadc(lkkm, 8)
        cr = broadc(craft, 8)
        combine = add_scores(cr, lk)
        hrts_d[lbl] = combine
    hrts_d["RoadAccidents/021"] = sims_d["RoadAccidents/021"]        
    return hrts_d
    
def smoothing(values_d, wdw):
    smooth_d = {}
    for lbl, val in values_d.items():
        c = []
        for i in range(len(val)):
            if i > wdw and (len(val)-i-1) > wdw:
                agg = sum(val[i-wdw:i+wdw])
                agg /= wdw
            elif i < wdw:
                agg = sum(val[:i])
                agg /= max(1,i)
            elif (len(val)-i-1) < wdw:
                agg = sum(val[i:])
                agg /= max(1,i)
            c.append(agg)
        smooth_d[lbl] = c
    return smooth_d

def score_combine(scores_d, sims_d, lkkm_d):
    
    #SIMS
    # sims = relative_scale(sims, 150, 0.5)
    # sims = delta_multiplier(sims, deltas_d)
    # sims = peak_window(sims, 20)
    # sims = center_on_avg_err(sims)
    # sims = bulkrangeto1(sims, 2, 1)
    
    #MIL
    #LKKM
    # lkkm_d = decompose(lkkm_d)
    # lkkm_d = relative_scale(lkkm_d, 50, 0.5)
    # lkkm_d = peak_window(lkkm_d, 20)
    # lkkm_d = center_on_avg_err(lkkm_d)
    # lkkm_d = bulkrangeto1(lkkm_d, 2, 1)
    

    combine_d = {}
    for lbl, lkkm in lkkm_d.items():
        craft = sims_d[lbl]
        if len(lkkm) > len(craft):
            x = np.asarray(craft)
            x.resize(len(lkkm))
            y = np.asarray(lkkm, refcheck=False)
        else:
            x = np.asarray(lkkm)
            x.resize(len(craft), refcheck=False)
            y = np.asarray(craft)
        combine = np.add(x, y)
        # print(combine)
        combine_d[lbl] = combine.tolist()
    # combine_d["RoadAccidents/021"] = sims_d["RoadAccidents/021"]        
    profile_d = consensus(combine_d, scores_d, wdw=6)
    
    profile_d = center_on_avg_err(profile_d)
    profile_d = bulkrangeto1(profile_d, 3, 0)

    return profile_d
                       
   
def anno_vs_score_eval(antn, scores_d):
    collect_cm = {}
    collect_roc = {}
    
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # thresholds = [x+0.015 for x in thresholds]
    
    for thresh in thresholds:
        anom_instances = []
        norm_instances = []
        anom_res = []
        norm_res = []
        anno_sgm_d = {}

        for lbl, scores in scores_d.items():
            scores = [float(i) for i in scores]
            cat, num = lbl.split('/')
            if not "Normal" in cat:
                vid_anno = antn[antn[:, 0] == cat]
                vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
                if len(vid_anno) == 0:
                    pass
                else:
                    n_frames = int(vid_anno[2])
                    anno_cum = []
                    res = False
                    for pt in range(3, len(vid_anno), 2):
                        if vid_anno[pt] != '-1':
                            start_anom = int(vid_anno[pt])
                            end_anom = int(vid_anno[pt+1])
                            res, anno_sgm = verify_anom_fc6(scores, n_frames, ((start_anom, end_anom)), thresh, prev_res=res)   
                            
                            anno_cum += anno_sgm
                            anom_instances.append(1) #every anom is an instance even if 2 anoms come from same video
                            if res: 
                                anom_res.append(1) 
                            else:
                                anom_res.append(0)
                        else:
                            print("-1 found, skip next -1")
                            pass
                    anno_sgm_d[lbl] = anno_cum
                    
            else:
                norm_instances.append(1)
                violating = list(filter(lambda x: (x > thresh), scores)) #maybe use filter instead
                if (len(violating) == 0):
                    norm_res.append(1) 
                else: 
                    norm_res.append(0)
                anno_sgm_d[lbl] = np.zeros(len(scores)).tolist()


            #pos is anomalous
            #neg is normal    
            
        tp = sum(anom_res)/sum(anom_instances)
        fn = 1 - tp
        tn = sum(norm_res)/sum(norm_instances) #TODO change back to normal instances
        fp = 1 - tn
        cm = np.asarray([[tp, fn],[fp, tn]])
        collect_cm[thresh] = cm
        
        # instances = anom_instances + norm_instances
        # results = anom_res + norm_res
        
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        collect_roc[thresh] = (tpr, fpr)
    
    # CALCULATE RES AT BEST THRESHOLD (for display of S or F marker)
    best_thresh = 0
    best_cm_sum = 0
    for t, mat in collect_cm.items():
        cm_sum = mat[0][0] + mat[1][1]
        if cm_sum > best_cm_sum:
            best_cm_sum = cm_sum
            best_thresh = t 
        
    print("BEST THRESHOLD: {}\ngetting result markers at that thresh ...".format(best_thresh))
    res_d = {}
    for lbl, scores in scores_d.items(): 
        scores = [float(i) for i in scores]
        cat, num = lbl.split('/')
        if not "Normal" in cat:
            # print("\t[PRED EVAL] anomalous video")
            vid_anno = antn[antn[:, 0] == cat]
            vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
            # print(vid_anno)
            if len(vid_anno) == 0:
                print("\t\tWARNING: annotation not found for {}/{}".format(cat, num))
                res_d[lbl] = None
            else:
                n_frames = int(vid_anno[2])
                res = False
                for pt in range(3, len(vid_anno), 2):
                    if vid_anno[pt] != '-1':
                        start_anom = int(vid_anno[pt])
                        end_anom = int(vid_anno[pt+1])
                        res, anno_sgm = verify_anom_fc6(scores, n_frames, ((start_anom, end_anom)), best_thresh, prev_res=res)   
                    else:
                        pass
                
                res_d[lbl] = res
        else:
            norm_instances.append(1)
            violating = list(filter(lambda x: (x > best_thresh), scores)) #maybe use filter instead
            if (len(violating) == 0): 
                norm_res.append(1) 
            else:
                norm_res.append(0)
            res_d[lbl] = (len(violating)==0)

    
    plt.rcParams['text.color'] = "black"
    plt.rcParams['axes.labelcolor'] = "black"
    plt.rcParams['xtick.color'] = "black"
    plt.rcParams['ytick.color'] = "black"
    plt.rcParams['axes.facecolor']='ffffff'
    plt.rcParams['savefig.facecolor']='ffffff'
    plt.rcParams.update({'font.size': 14})

    
    cnt = 0
    r = 0
    figcm, axcm = plt.subplots(3,3,constrained_layout=True)
    for t, cmat in collect_cm.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=CLASSES)
        disp.plot(cmap='plasma', ax=axcm[r][cnt % 3], colorbar=False) 
        axcm[r][cnt%3].title.set_text("threshold = {}".format(t))
        cnt+=1
        if cnt % 3 == 0:
            r += 1
            
    figbestcm, axbestcm = plt.subplots(1,1)
    dispbest = ConfusionMatrixDisplay(confusion_matrix=collect_cm[0.4], display_labels=CLASSES)
    dispbest.plot(cmap='Blues', ax=axbestcm, colorbar=True)
    
    return figbestcm, anno_sgm_d, res_d


def st_demo(combine_d, scores_d, sims_d, lkkm_d, anno_sgm_d, res_d, cm_disp, roc_disp):
    st.markdown("<h1 style='text-align: center; color: white; font-size: 2.5vw'>Anomaly Detection Framework</h1>", unsafe_allow_html=True)
    st.markdown("### This demo displays results of the application of an anomaly detection framework to a set of unseen CCTV surveillance videos.")
    st.markdown("### The framework assigns an anomaly score in the range [0, 1] per 16 frames of video.")
    st.markdown("### The final score is decided as a consensus between 3 models/heuristics, namely: *MIL base-model*, *CRAFT-flow*, *LKKM-flow*")
    st.markdown("### At each video, the display of relevant scores can be toggled and an option to view the video is presented.")
    st.markdown("### The categories of video include:")
    """
    * Abuse
    * Arrest
    * Arson
    * Assault
    * Burgalry
    * Explosion
    * Fighting
    * Road Accidents 
    * Robbery
    * Shooting
    * Shoplifting
    * Stealing
    * Vandalism
    * Normal
    """
    
    st.image(roc_disp, width=None)
    st.image(cm_disp, width=None)
    
    cnt = 0
    
    
    for vid in sorted(os.listdir(VIDEO_PATH), reverse=True):
        # st.write(vid)
        cnt += 1
        if vid.endswith(".mp4"):
            title = os.path.splitext(vid)[0][:-5]
            
            if title[:6] == "Normal":
                cat = "Normal"
            else:
                cat = title[:-3]
            num = title[-3:]

            lbl = "{}/{}".format(cat, num)
            
            if lbl in res_d.keys():
                if res_d[lbl] is None:
                    emj = ":white_circle:" 
                else:
                    emj = ":large_blue_circle:" if res_d[lbl] else ":red_circle:" 
                    
            exp = st.expander(label=lbl, expanded=False)
            with exp:
                if exp.checkbox("Playback Video", key=lbl, value=False):
                    if cat == "Normal":
                        path_mp4 = os.path.join(VIDEO_PATH, "{}_Videos_{}_x264.mp4".format(cat,num))
                    else:
                        path_mp4 = os.path.join(VIDEO_PATH, "{}{}_x264.mp4".format(cat,num))
                           
                    vf = open(path_mp4, "rb")
                    vb = vf.read()
                    st.video(vb)
                    
                if lbl in combine_d.keys():
                    combine_data = pd.DataFrame({
                        "final scores": combine_d[lbl],
                    })
                else:
                    combine_data = pd.DataFrame({
                        "final scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in COMBINE dict.")
                
                if lbl in sims_d.keys():
                    flow_data = pd.DataFrame({
                        "flow scores": sims_d[lbl],
                    })
                else:
                    flow_data = pd.DataFrame({
                        "flow scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in FLOW dict.")
                    
                if lbl in lkkm_d.keys():
                    lkkm_data = pd.DataFrame({
                        "lkkm scores": lkkm_d[lbl],
                    })
                else:
                    lkkm_data = pd.DataFrame({
                        "lkkm scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in LKKM dict.")
                    
                if lbl in scores_d.keys():
                    mil_data = pd.DataFrame({
                        "MIL scores": scores_d[lbl],
                    })
                else:
                    mil_data = pd.DataFrame({
                        "MIL scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] WARNING no matching label for this video in MIL scores dict.")    
                    
                res = None
                anno_dat = np.zeros(len(scores_d[lbl])).tolist()
                if lbl in anno_sgm_d.keys():
                    anno_sgms = anno_sgm_d[lbl]
                    if cat != "Normal":
                        for a in anno_sgms:
                           anno_dat[int(a)] = 1 #1 indexing

                        set1 = False
                        for a in range(len(anno_dat)):
                            if not set1:
                                if anno_dat[int(a)] == 1:
                                    set1 = True
                            else:
                                if anno_dat[int(a)] == 1:
                                    set1 = False
                                else:
                                    anno_dat[int(a)] = 1
                    else:
                        anno_dat[0] = 0 #to ensure that y-ax is scaled to 1 for normal footage
                        
                    anno_data = pd.DataFrame({
                        "annotations": anno_dat,
                    })
                else:
                    anno_data = pd.DataFrame({
                        "annotations": anno_dat,
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in ANNO dict.")
                
                all_data = anno_data 
                
                st.checkbox("Display MIL", value=False, key="{}{}MIL".format(cat, num)) 
                st.checkbox("Display CRAFT", value=False, key="{}{}FLOW".format(cat, num))
                st.checkbox("Display LKKM", value=False, key="{}{}LKKM".format(cat, num))
                st.checkbox("Display FINAL", value=True, key="{}{}FINAL".format(cat, num))
    
                if eval("st.session_state.{}{}MIL".format(cat, num)):
                    all_data["base-score"] = mil_data
                if eval("st.session_state.{}{}FLOW".format(cat, num)):
                    all_data["craft-score"] = flow_data
                if eval("st.session_state.{}{}LKKM".format(cat, num)):
                    all_data["lkkm-score"] = lkkm_data
                if eval("st.session_state.{}{}FINAL".format(cat, num)):
                    all_data["consensus"] = combine_data

                st.area_chart(all_data, use_container_width=True)
                

def consensus(base, sims, lkkm, wdw):
    profile_d = {}
    for lbl, b in base.items():
        s = sims[lbl]
        if lbl == "RoadAccidents/021":
            l = sims[lbl]
        else:
            l = lkkm[lbl]
        c = []
        assert len(s) == 32
        assert len(b) == 32
        assert len(l) == 32
        
        for i in range(32):
            if b[i] == 0:
                c.append(b[i]) 
            else:
                c.append(max([b[i], s[i], l[i]]))
                # c.append(0.5*(1+h[i])*b[i])
        profile_d[lbl] = c
        assert len(c) == 32
    return profile_d    

def main(pred_path, sim_path, lkkm_path):
    antn = read_anno_file(ANNO_FILE)
    with open(pred_path, "r") as f:
        scores = json.load(f)
        scores_d = json.loads(scores)
    with open(sim_path, "r") as f:
        sims = json.load(f)
        sims_d = json.loads(sims)
    with open(lkkm_path, "r") as f:
        lkkm = json.load(f)
        lkkm_d = json.loads(lkkm)

    
    #CRAFT process
    sims_d = convert_to_32(sims_d)
    lkkm_d = decompose(lkkm_d)
    lkkm_d = convert_to_32(lkkm_d)
    
    #indiv process
    sims_d = individual_scale(sims_d)
    sims_d = filter_to_zero(sims_d, 1.5)
    sims_d = nan_to_zero(sims_d)
    sims_d = to_range(sims_d, 1)
    sims_d = apply_broadc(sims_d, 16)
    
    lkkm_d = individual_scale(lkkm_d)
    lkkm_d = filter_to_zero(lkkm_d, 1.5)
    lkkm_d = nan_to_zero(lkkm_d)
    lkkm_d = to_range(lkkm_d, 1)
    lkkm_d = apply_broadc(lkkm_d, 16)
    
    scores_d = nan_to_zero(scores_d)
    scores_d = filter_to_zero(scores_d, 0.3)
    scores_d = apply_broadc(scores_d, 16)
    
    profile_d = consensus(scores_d, sims_d, lkkm_d, 1)

    #SEQUENCE
    # anno_seq, score_seq = compute_score_seq(scores_d, antn)
    # anno_seqp, profile_seq = compute_score_seq(profile_d, antn)
    # anno_seql, lkkm_seq = compute_score_seq(lkkm_d, antn)
    # anno_seqc, sims_seq = compute_score_seq(sims_d, antn)
    
    #ROC/AUC
    # figroc, ax = plt.subplots(1,1)
    # 
    # ax.plot([0,1], [0,1], label='Binary SVM: AUC=0.5', linestyle="dashed", color="black", linewidth=3)
    # 
    # fprb, tprb, threshb = roc_curve(anno_seq, score_seq)
    # aucb = roc_auc_score(anno_seq, score_seq)
    # 
    # ax.plot(fprb,tprb,label="Base: AUC="+str(round(aucb,4)), color="blue",linewidth=3)
    # 
    # fprc, tprc, threshc = roc_curve(anno_seqc, sims_seq)
    # aucc = roc_auc_score(anno_seqc, sims_seq)
    # ax.plot(fprc,tprc,label="CRAFT: AUC="+str(round(aucc,4)), color="purple", linewidth=3)
# 
    # fprl, tprl, threshl = roc_curve(anno_seql, lkkm_seq)
    # aucl = roc_auc_score(anno_seql, lkkm_seq)
    # ax.plot(fprl,tprl,label="LKKM: AUC="+str(round(aucl,4)), color="orange", linewidth=3)
    # 
    # fprp, tprp, threshp = roc_curve(anno_seqp, profile_seq)
    # aucp = roc_auc_score(anno_seqp, profile_seq)
    # ax.plot(fprp,tprp,label="Consensus: AUC="+str(round(aucp,4)), color="red",linewidth=3)
    # 
    # plt.legend(loc=0, fontsize=13)
    # plt.xlabel('False Positive Rate', fontsize=14)
    # plt.ylabel('True Positive Rate', fontsize=14)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)    
    # plt.show()
    
    # ConfusionMatrixDisplay.from_predictions(anno_seq, score_seq, normalize='true')
    # plt.show()
    
    # thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # thresholds = [x+0.05 for x in thresholds]

    # for thresh in thresholds:
    #     score_seq_class = []
    #     for i in range(len(profile_seq)):
    #         if profile_seq[i] > thresh:
    #             score_seq_class.append(1)
    #         else:
    #             score_seq_class.append(0)
    #     print(f"THRESH: {thresh}")
    #     print(anno_seq)
    #     print(score_seq)
    #     print(score_seq_class)     
    #     ConfusionMatrixDisplay.from_predictions(anno_seq, score_seq_class, normalize='true')
    #     plt.show()
    
    
    cm_disp, anno_sgm_d, res_d = anno_vs_score_eval(antn, scores_d) #TODO back to profile
    cm_img =  Image.open("demo/img/CMsb.png")
    roc_img =  Image.open("demo/img/ROCsb.png")


    
    st_demo(combine_d=profile_d, #TODO should be profile_d
        scores_d=scores_d,
        sims_d=sims_d,
        lkkm_d=lkkm_d,
        anno_sgm_d=anno_sgm_d,
        res_d=res_d,
        cm_disp=cm_img,
        roc_disp=roc_img)
        
if __name__ == '__main__':
    pred_path = sys.argv[1]
    sim_path = sys.argv[2]
    lkkm_path = sys.argv[3]
   
    main(pred_path, sim_path, lkkm_path)
