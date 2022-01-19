from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from math import ceil
import numpy as np

normal_out = ["Normal/059", "Normal/129","Normal/189","Normal/478","Normal/656","Normal/877","Normal/887","Normal/898","Normal/901","Normal/925",]

def by_score(obj):
    return obj[0]

def compute_score_seq(scores_d, antn):
    anno_seq = []
    score_seq = []
    normal_maxis = []
    for lbl, scores in scores_d.items():
        scores = [float(i) for i in scores]
        cat, num = lbl.split('/')

        n_scores = 32
        anno_profile = np.zeros(n_scores)
        split_indices = []
        
        # ANOMALOUS VIDEOS
        if not "Normal" in cat:
            # GET ANNOTATIONS
            vid_anno = antn[antn[:, 0] == cat]
            vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
            
            if len(vid_anno) == 0:
                pass
            else:
                n_frames = int(vid_anno[2])
                for pt in range(3, len(vid_anno), 2):
                    if vid_anno[pt] != "-1":
                        start_anom = int(vid_anno[pt])
                        end_anom = int(vid_anno[pt+1])
                        start_anom = max(0, int((start_anom/n_frames)*n_scores - 4))
                        end_anom = min(n_scores, ceil((end_anom/n_frames)*n_scores + 4))
                        for i in range(start_anom, end_anom):
                            anno_profile[i] = 1
                            
                        split_indices.append(max(0,start_anom-1))
                        split_indices.append(end_anom)

            split_scores = np.array_split(scores, split_indices)
            split_annos = np.array_split(anno_profile, split_indices)

            for sc in split_scores:
                if len(sc.tolist()) != 0:
                    score_seq.append(max(sc))
            for sc in split_annos:
                if len(sc.tolist()) != 0:
                    anno_seq.append(max(sc))
                    
        else:
            if not lbl in normal_out:
                score_seq.append(max(scores))
                anno_seq.append(0)
            
    return anno_seq, score_seq
    
    
def compute_ROC(anno_seq, score_seq):
    RocCurveDisplay.from_predictions(anno_seq, score_seq)
    plt.show()
    