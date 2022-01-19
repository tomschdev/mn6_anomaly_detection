from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
from keras.models import model_from_json
from functools import reduce
from random import shuffle
import copy
import time
from itertools import islice
import json
from datetime import datetime
from scipy.io import loadmat, savemat
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings
warnings.filterwarnings('ignore')

#CHECKLIST:
# - loading correct weights
# - using correct function to load weights
# - pointing to correct features
# - naming prediction path correctly

FEAT="c3d/feature"
BATCH = 30
LAMBDA2 = 0.00008 #sparsity
LAMBDA3 = 0.00008 #temporal smoothness

root = os.getcwd()
model_store = os.path.join(root, "annMIL/model")
x = datetime.now()
model_path = os.path.join(model_store, "arch_8")
weight_path = os.path.join(model_store, "weights", "thenocrossarch_E8000", "best_auc_model_vthenocrossarch_E8000_full_allepoch")
pred_path = os.path.join(root, "annMIL/score/demo", "pred_{}.json".format(os.path.basename(weight_path)))
  
"""
METRICS & LOSS
"""
# VERIFIED
def max_norm_scores_metric(y_true, y_pred):
    """
    Provision of average difference between max scores in normal bag for a batch
    """
    norm_preds = y_pred[BATCH*32:]
    assert len(norm_preds) == (BATCH*32), "max norm scores is operating on too many bag instances"
    max_scores_norm_bags = []
    for pair_num in range(BATCH): #goes through pairs in batch of 30 and gets loss for each
        norm_scores = norm_preds[pair_num*32:(pair_num+1)*32]
        max_scores_norm_bags.append(tf.math.reduce_max(norm_scores))
    max_scores_norm_bags = tf.stack(max_scores_norm_bags)
    mean_max_norm = tf.math.reduce_mean(max_scores_norm_bags)
    return mean_max_norm 

# VERIFIED
def max_anom_scores_metric(y_true, y_pred):
    """
    Provision of average difference between max scores in anomalous bag for a batch
    """
    anom_preds = y_pred[:BATCH*32]
    max_scores_anom_bags = []
    for pair_num in range(BATCH): #goes through pairs in batch of 30 and gets loss for each
        anom_scores = anom_preds[pair_num*32:(pair_num+1)*32]
        max_scores_anom_bags.append(tf.math.reduce_max(anom_scores))
    max_scores_anom_bags = tf.stack(max_scores_anom_bags)    
    mean_max_anom = tf.math.reduce_mean(max_scores_anom_bags)
    return mean_max_anom 

# VERIFIED
def difference_of_avg_max(y_true, y_pred):
    mma = max_anom_scores_metric(y_true, y_pred)
    mmn = max_norm_scores_metric(y_true, y_pred)
    df = tf.math.subtract(mma, mmn)
    return df

# VERIFIED
def compute_AVLE(y_true, y_pred, T): 
    anom_pred = y_pred[:BATCH*32]

    y_class = tf.greater(anom_pred, T)
    y_class = tf.cast(y_class, tf.int32)
    
    total_instances_anom = len(anom_pred)
    
    anom_labels = tf.ones((total_instances_anom, 1), tf.int32)
    eval_anom = tf.math.abs(tf.math.subtract(y_class, anom_labels))

    anom_success = []
    for pair_num in range(BATCH):
        anom_success.append(tf.cast(
            tf.math.not_equal(
                tf.math.reduce_sum(eval_anom[pair_num*32:(pair_num+1)*32]), tf.constant(32, tf.int32)), dtype=tf.int32))
    
    return tf.math.divide(
            tf.cast(tf.math.accumulate_n(anom_success), dtype=tf.int32), 
            tf.cast(tf.math.divide(tf.cast(BATCH, dtype=tf.int32), tf.constant(32, tf.int32)), dtype=tf.int32)) 
    
# VERIFIED
def compute_NVLE(y_true, y_pred, T):  
    
    norm_pred = y_pred[:BATCH*32]

    y_class = tf.greater(norm_pred, T)
    y_class = tf.cast(y_class, tf.int32)

    total_instances_norm = len(norm_pred)
    
    norm_labels = tf.zeros((total_instances_norm, 1), tf.int32)
    eval_norm = tf.math.abs(tf.math.subtract(y_class, norm_labels))

    norm_success = []
    for pair_num in range(BATCH):
        norm_success.append(tf.cast(
            tf.math.equal(
                tf.math.reduce_sum(eval_norm[pair_num*32:(pair_num+1)*32]), tf.constant(0, tf.int32)), dtype=tf.int32))
    
    return tf.math.divide(
            tf.cast(tf.math.accumulate_n(norm_success), dtype=tf.int32), 
            tf.constant(BATCH, dtype=tf.int32))

# VERIFIED
def auc_VLE(y_true, y_pred):
    t_range = np.arange(0, 1, 0.05, dtype=np.int32)
    fprs = []
    tprs = []
    print("[video-level con-mats]")
    for t in t_range:
        tp = compute_AVLE(y_true, y_pred, t)
        fn = 1 - tp
        tn = compute_NVLE(y_true, y_pred, t) 
        fp = 1 - tn
        cmat = [[tp, fp], [fn, tn]]
        print(cmat)
        fpr = fp/(fp+tn)
        fprs.append(fpr)
        tpr = tp/(tp+fn)
        tprs.append(tpr)
    aucs = auc(fprs, tprs)
    return aucs

# VERIFIED
def tf_auc_VLE(y_true, y_pred):
    m = None
    m = tf.keras.metrics.AUC(num_thresholds=20)
    m.update_state(y_true, y_pred)
    return tf.cast(m.result(), dtype=tf.float32)
    
# VERIFIED
def ranking_loss(y_true, y_pred):
    """
    This dictates the loss function to be followed when training the ANN.
    The model makes use of a ranking loss which allows the network to learn the difference between high-scoring normal activity and
    high-scoring anomalous activity.
    
    @param y_true: labels of video segments of mini-batch - 1:anomalous 0:normal
    @param y_pred: predicted scores of video segments of mini-batch
    
    Returns
    net_loss: net loss of batch computed by summing losses per pair of anomalous and normal bags
    """
    #assume y_true is same as 'targets' i.e., the exact label tensor passed in for training 
    #a label per sgm_vec per video - for 60 videos
    anom_preds = y_pred[:BATCH*32]
    #print("len of anom_preds = {}".format(len(anom_preds)))
    norm_preds = y_pred[BATCH*32:BATCH*32*2]
    #print("len of norm_preds = {}".format(len(norm_preds)))
    
    batch_losses = []
    for pair_num in range(BATCH): #goes through pairs in batch of 30 and gets loss for each

        anom_scores = anom_preds[pair_num*32:(pair_num+1)*32]
        norm_scores = norm_preds[pair_num*32:(pair_num+1)*32]

        max_anom = tf.math.reduce_max(anom_scores)
        max_norm = tf.math.reduce_max(norm_scores)
        # min_anom = tf.math.reduce_min(anom_scores) #extension constraint
        # min_max_diff = tf.math.subtract(max_anom, min_anom)
        # min_diff_penalty = tf.math.subtract(anom_scores, min_anom)

        temporal = tf.math.reduce_sum(tf.math.square(tf.experimental.numpy.diff(anom_scores)))
        sparsity = tf.math.reduce_sum(anom_scores)
      
        elems = [tf.math.negative(max_anom), max_norm, tf.constant(1, dtype="float32")]
        expr = tf.maximum(tf.constant(0, dtype="float32"), tf.math.reduce_sum(elems))
      
        const_elems = [expr, tf.math.multiply(LAMBDA2, sparsity), tf.math.multiply(LAMBDA3, temporal)]
        # tf.math.multiply(LAMBDA1, tf.math.negative(min_max_diff)) # took this out of above statement - extension
        constr_expr = tf.math.reduce_sum(const_elems) #this is the loss for one of the 30 pairs in the batch
      
        batch_losses.append(constr_expr)
    batch_losses = tf.stack(batch_losses)
    mean_batch_loss = tf.math.reduce_mean(batch_losses)
    return mean_batch_loss #NB note regularization ||W|| is applied automatically after feeding kernel_reg param in model construction
"""
END METRICS & LOSS
"""

'''
UTILITIES
'''
def string_util(x, y):
    return "{}\n{}\n".format(x,y)

def bynum(x):
    return int(os.path.splitext(x)[0])

def bysgm(x):
    if x == "sgm_avg":
        return 9999999
    else:
        return int(x.split("_")[1])

def decode_fc6(fc6):
    with open(fc6, 'rb') as f:
        elements = 5                #: num, chanel, length, height, width
        element_size_byte = 4       # 32bit = 4byte
        total_header_size = elements*element_size_byte
        # The width and height are 4 bytes each, so read 8 bytes to get both of them
        header_bytes = f.read(total_header_size)
        # we decode the byte array from the last step.
        header = struct.unpack('i' * elements, header_bytes)
        data_size = header[0]*header[1]*header[2]*header[3]*header[4]
        data_bytes = f.read(element_size_byte*data_size)
        data = struct.unpack('f' * data_size, data_bytes)
        return np.array(data)
    
def read_all_fc6(path):
    """
    for reading in fc6 files per 16 frames of a video
    """
    print(path)
    all_fc6 = []
    for sgms in sorted(os.listdir(path), key=bysgm):
        if "avg" not in sgms:
            for fl in sorted(os.listdir(os.path.join(path, sgms)), key=bynum):
                data = decode_fc6(os.path.join(path, sgms, fl))
                data = np.reshape(np.asarray(data).astype("float32"), ((1, 4096)))
                all_fc6.append(data)
        
    all_data = np.reshape(np.asarray(all_fc6).astype("float32"), ((len(all_fc6), 4096)))
    print("shape of all fc6s in vid:", all_data.shape)
    return all_data

def read_all_fc6_lp(path):
    """
    for reading in fc6 files per 16 frames of a video
    note: this function assumes that fc6 files are not sorted into segments!
    """
    print(path)
    all_fc6 = []

    for fl in sorted(os.listdir(os.path.join(path)), key=bynum):
        # print(fl)
        # data = decode_fc6(os.path.join(path, sgms, fl))
        with open(fl, "r") as fc6:
            data = np.reshape(np.asarray(fc6).astype("float32"), ((1, 4096)))
            all_fc6.append(data)
        # f.close()
        
    all_data = np.reshape(np.asarray(all_fc6).astype("float32"), ((len(all_fc6), 4096)))
    print("shape of all fc6s in vid:", all_data.shape)
    return all_data

def read_mono(all_IDs):
    print("\t\t\t--->reading mono")
    print("len of paths: {} ".format(len(all_IDs)))

    mono = {}
    pairs = len(all_IDs)
    
    #collect random set of training paths from all normal and anomalous paths
    for path in all_IDs:
        filename = path.split("/")[-1:]
        cat, num = filename[:-3], filename[-3:]
        lbl = "{}/{}".format(cat, num)
        bag = read_video_stack(path) # 32x4096
        mono[lbl] = bag # a video's 4096D tensors in 2D tensor form 

    return mono 

def read_mono_video_to_32(path):
    with open(path, "r") as f:
        all_features = json.load(f)
        all_features_d = json.loads(all_features)
    
    lbl = all_features_d.keys()
    split_sgms = np.array_split(all_features_d.values(), 32)
    avgd_split_sgms = []
    for features in split_sgms:
        avgd_split_sgms.append(np.mean(features, axis=0))
    
    return all_features_d

def read_mono_video(path):
    with open(path, "r") as f:
        all_features = json.load(f)
        all_features_d = json.loads(all_features)
    
    return all_features_d

def read_mono_lp(path):
    """
    this function reads a file which contains a full dictionary of all videos features
    """
    with open(path, "r") as f:
        all_features = json.load(f)
        all_features_d = json.loads(all_features)
    
    return all_features_d

def read_video_stack(path):
    f= open(path, "r")
    lines = f.readlines()
    data = [l.strip().split(' ') for l in lines]
    data = np.reshape(np.asarray(data).astype("float32"), ((len(data), 4096)))
            
    return data

def gen_lists(c3d_vault):
    """
    Accumulate list of all paths to training videos - leading up to random selection
    """
    print("[TEST_NET] fetching test data paths")
    
    feat_paths = []
    feat_dir = os.listdir(c3d_vault)
    for path in feat_dir:
        print("path: {}".format(path))
        concat_path = os.path.join(c3d_vault, path)
        feat_paths.append(concat_path)
                              
    # print all paths                 
    # print(reduce(string_util, norm_video_paths))
    # print(reduce(string_util, anom_video_paths))
    return feat_paths

def tf_auc_VLE(y_true, y_pred):
    m = None
    m = tf.keras.metrics.AUC(num_thresholds=20)
    m.update_state(y_true, y_pred)
    return tf.cast(m.result(), dtype=tf.float32)

'''
END UTILITIES
'''


"""
WAQAS WEIGHTS

to use waqas weights,
    change weights path to point to waq_weights
    make change in load_trained_model to use load_weights from here, not from keras
"""
def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

"""
END WAQAS WEIGHTS
"""

def load_trained_model():
    """
    @param model_path - path to model architecture and weights
    """
    model = tf.keras.models.load_model(model_path, custom_objects={
            "ranking_loss": ranking_loss,
            "max_norm_scores_metric": max_norm_scores_metric,
            "max_anom_scores_metric": max_anom_scores_metric,
            "difference_of_avg_max": difference_of_avg_max,
            "tf_auc_VLE": tf_auc_VLE,
        })
    model.load_weights(weight_path)
    print("[TEST_NET] model loaded:\n\tarch at: {}\n\tweights at: {}".format(model_path, weight_path))
    return model
    
def main():
    model = load_trained_model()
    test_paths = gen_lists(FEAT)

    pred_mono = {}
    for video in test_paths:
        filename = video.split(".")[0]
        filename = filename.split("/")[-1:][0]
        cat, num = filename[:-3], filename[-3:]
        lbl = "{}/{}".format(cat, num)
        print(filename)
        bag = read_video_stack(video) # 32x4096
        print("\t\t\t{} x {} -- bag shape".format(bag.shape[0], bag.shape[1]))
        
        l_pred = model.predict(bag) 
        l_pred = [round(float(np.squeeze(x)), 4) for x in l_pred]
        pred_mono[filename] = l_pred
    
    pred_dict = json.dumps(pred_mono)
    print("[TEST_NET] predictions produced from loaded model")
    
    fig, ax = plt.subplots(1, len(pred_mono), squeeze=False)
    cnt = 0
    for v, s in pred_mono.items():
        ax[0, cnt].plot(np.arange(len(s)),s)
        ax[0, cnt].set_xlabel("SCORE PROFILE: " + v)
        cnt+=1
    plt.show()

    with open(pred_path, "w") as jf:
        print("[TEST_NET] predictions written to {}".format(pred_path))
        json.dump(pred_dict, jf)
        
if __name__ == '__main__':
    main()  