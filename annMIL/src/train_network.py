import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l2
import numpy as np
import os
from scipy.io import loadmat, savemat
from keras.models import model_from_json
from functools import reduce
from random import shuffle
import copy
import time
from itertools import islice
from keras.callbacks import ModelCheckpoint
import json
from sklearn.metrics import auc
import pandas as pd

#CHECKLIST
# check which weights are referenced at WEIGHT_PATH - None for random initial
# check FREEZE toggle
# check version - 'rdm', 'tune', 'arch', 'constraint' NOTE: patience auto appended
# check parameters 
# check record store
# check patience
# check that there is a time dir, record dir, model dir, anom_path.txt, norm_path.txt

root = os.getcwd()
MODEL_STORE = os.path.join(root, 'model')
RECORD_STORE = os.path.join(root, 'record')
TIME_STORE = os.path.join(root, 'time')
WEIGHT_PATH = os.path.join(root, "waqas", "weights_L1L2.mat")
WEIGHT_PATH = None
FREEZE = False
TRANSFER_LEARN = False

f=open(os.path.join(root, "anom_path.txt"), "r")
ANOM_FEAT=f.readline().strip()
f.close()
f=open(os.path.join(root, "norm_path.txt"), "r")
NORM_FEAT=f.readline().strip()
f.close()

# VERIFIED
BATCH = 30
TEST_BATCH = 30
N_FOLDS = 8
EPOCH = 7000
# LAMBDA1 = float(10.0) #EXTENSION
LAMBDA2 = 0.00008 #sparsity
LAMBDA3 = 0.00008 #temporal smoothness
DROPOUT = 0.4 #waqas uses 0.6 as probability of retain, keras wants probability of drop -> therefore 0.4
THRESHOLD = 0.5
VERSION = "rdm"
LR = 0.001
PATIENCE=10 # choose patience from [5,10,20,40]
FREQ=10

VERSION = VERSION + "_P" + str(PATIENCE*FREQ)


#VERIFIED
class EarlyStopLoss(keras.callbacks.Callback):
    """
    Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
    """

    def __init__(self, patience=PATIENCE): #NB note that this is the number of validations, each validation implies 20 training epochs 
        super(EarlyStopLoss, self).__init__()
        self.patience = patience
        self.best_weights = None # best_weights to store the weights at which the minimum loss occurs.

    def on_train_begin(self, logs=None):
        self.wait = 0 # number of epoch waited while loss is no longer minimum.
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        toprint=False
        for k in logs.keys():
            if "val" in k:
                toprint=True
                
        if toprint:
            current_val = logs.get("val_loss")
            current_train = logs.get("loss")
            current_ratio = float(current_val/current_train)
            if np.less(current_ratio, self.best):
                self.best = current_ratio
                self.wait = 0
                self.best_weights = self.model.get_weights() # record the best weights if current loss is less
            else:
                self.wait += 1
                print("PATEINCE: {}/{}".format(self.wait, self.patience))
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    #print("Restoring model weights from the end of the best epoch (based on validation loss/training loss).")
                    #fold = 1
                    #self.model.set_weights(self.best_weights)
                    #while os.path.exists(os.path.join(MODEL_STORE, VERSION, "best_val_loss_ratio_model_v{}_fold{}.index".format(VERSION, fold))):
                    #    fold+=1
                    #self.model.save_weights(os.path.join(MODEL_STORE, VERSION, "best_val_loss_ratio_model_v{}_fold{}".format(VERSION, fold)))


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

#VERIFIED
class OutputCall(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        toprint=False   
        for k in logs.keys():
            if "val" in k:
                toprint=True
            
        if toprint:
            print("val metrics identified:")
            print("--------------------------------------*EPOCH {}".format(epoch))  
            print("val/train ratio: {}".format(float(logs["val_loss"]/logs["loss"])))
            for k, v in logs.items():
                print("{}: {}".format(k, v))        
            print("--------------------------------------*")        

#VERIFIED
class OutputLogJSON(keras.callbacks.Callback):
    def __init__(self, meta):
        super(OutputLogJSON, self).__init__()
        self.t_params = meta["t_params"]
        self.fold = meta["fold"]
        print("FOLD: {}".format(meta["fold"]))
        self.json_log = None
        self.write_eval_logs = False
    
    def on_train_begin(self, logs):
        self.json_log = open(os.path.join(self.t_params["record_store"], 'log_v{}_fold{}.json'.format(self.t_params["version"], self.fold)), mode='wt', buffering=1)
        self.json_log.write(
            json.dumps({
                "num_epoch": str(self.t_params["num_epoch"]),
                "batch": str(self.t_params["batch"]),
                "test_batch": str(self.t_params["test_batch"]),
                "lambda_1": "extension: not currently used",
                "lambda_2": str(self.t_params["lambda_2"]),
                "lambda_3": str(self.t_params["lambda_3"]),
                "LR": str(self.t_params["LR"]),
                "dropout": str(self.t_params["dropout"]),
                "optimizer": str(self.t_params["optimizer"]),
                "version": str(self.t_params["version"]),
                "patience": str(PATIENCE) + " rounds of validation - val_freq={}".format(FREQ),
                "fold": str(self.fold+1),                
                }) 
            )
        self.json_log.write("\n")
        
    def on_train_end(self, logs):
        self.write_eval_logs = True
      
    def on_epoch_end(self, epoch, logs):
        toprint=False
        for k in logs.keys():
            if "val" in k:
                toprint=True
                
        if toprint:
            self.json_log.write(
                json.dumps({"epoch": epoch,
                    "val_loss": logs["val_loss"],
                    "val/train ratio": float(logs["val_loss"]/logs["loss"]),
                    "val_auc_VLE": logs["val_tf_auc_VLE"],
                    "val_max_anom_scores_metric": logs["val_max_anom_scores_metric"],
                    "val_max_norm_scores_metric": logs["val_max_norm_scores_metric"],
                    "loss": logs["loss"],
                    "auc_VLE": logs["tf_auc_VLE"],
                    "max_anom_scores_metric": logs["max_anom_scores_metric"],
                    "max_norm_scores_metric": logs["max_norm_scores_metric"],
                    }) 
                )
            self.json_log.write("\n")
            
    def on_test_end(self, logs):
        if self.write_eval_logs:
            # print(logs.keys())
            self.json_log.write("FINAL FOLD EVALUATION\norder:\n1. best val loss\n2. best difference\n")
            self.json_log.write(
                    json.dumps({
                        "val_loss": logs["loss"],
                        "val_auc_VLE": logs["tf_auc_VLE"],
                        "val_max_anom_scores_metric": logs["max_anom_scores_metric"],
                        "val_max_norm_scores_metric": logs["max_norm_scores_metric"],
                        }) 
                    )
            self.json_log.write("\n")
            
#VERIFIED
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, anom_IDs, norm_IDs, all_labels, batch_size, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.all_labels = all_labels
        self.anom_IDs = anom_IDs
        self.norm_IDs = norm_IDs
        self.indices = np.arange(len(self.norm_IDs))
        self.mono = self.read_mono()

    def read_video_block(self, path):
        f= open(path, "r")
        lines = f.readlines()
        data = [l.strip().split(' ') for l in lines]
        data = np.reshape(np.asarray(data).astype("float32"), ((32, 4096)))
        return data
    
    def read_mono(self):
        print("\t\t\t[reading mono]")
        print("len of paths: {} ".format(len(self.anom_IDs)))
        print("len of paths: {} ".format(len(self.norm_IDs)))

        mono = {}
        pairs = len(self.norm_IDs)
        
        #collect random set of training paths from all normal and anomalous paths
        for paths in [self.anom_IDs, self.norm_IDs]:                
            for path in paths:
                #print("\tselected video:", end = ' ')
                #print("\t{}/{}".format(os.path.basename(os.path.dirname(path)), os.path.basename(path)))
                bag = self.read_video_block(path) # 32x4096
                mono[path] = bag #32 4096D tensors in 2D tensor form 

        return mono 

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print("__len__: {}".format(int(np.floor(len(self.norm_IDs) / self.batch_size))))
        return int(np.floor(len(self.norm_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        anom_IDs_temp = [self.anom_IDs[k] for k in indices]
        norm_IDs_temp = [self.norm_IDs[k] for k in indices]

        # Generate data
        X, y = self.__data_generation(anom_IDs_temp, norm_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        if self.shuffle == True:
            shuffle(self.anom_IDs)
            shuffle(self.norm_IDs)   
    
    def __data_generation(self, anom_IDs_temp, norm_IDs_temp):
        """
        Generates data containing batch_size samples
        X : (n_samples, *dim, n_channels)
        """
        # Initialization
        Xa = []
        ya = []
        Xn = []
        yn = []
        # select data from pre-loaded mono
        for i, ID in enumerate(anom_IDs_temp):
            Xa.append(self.mono[ID]) #32 x 4096 np array
            ya.append(np.ones((32, 1))) #32 x 1 np array 
            
        for i, ID in enumerate(norm_IDs_temp):
            Xn.append(self.mono[ID]) #32 x 4096 np array
            yn.append(np.zeros((32, 1))) #32 x 1 np array 

        Xa = np.reshape(Xa, [self.batch_size*32, 4096]) #stack all feature vectors from anom and norm videos
        ya = np.reshape(ya, [self.batch_size*32, 1]) #stack all corresponding labels of above feature vectors
        Xn = np.reshape(Xn, [self.batch_size*32, 4096]) #stack all feature vectors from anom and norm videos
        yn = np.reshape(yn, [self.batch_size*32, 1]) #stack all corresponding labels of above feature vectors
        X = np.vstack((Xa,Xn))
        y = np.vstack((ya,yn))
        #print("shapes of X, Xa, Xn")
        #print(Xa.shape)
        #print(Xn.shape)        
                
        return X, y 

'''
UTILITIES
'''
def string_util(x, y):
    return "{}\n{}\n".format(x,y)

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

def load_weights_waqas(model, weight_path):
    """
    Utility function for loading weights provided by Sultani et al.
    """
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def load_model(model_path):
    """
    This function loads the ANN created by save_ann_model() - therefore the function create_ann_model() is only required to run once and thereafter
    the model is saved and loaded from json.
    This function will be used both in testing and training.
    """
    annMIL = tf.keras.models.load_model(os.path.join(model_path, "best_model_*"))
    return annMIL

'''
END UTILITIES
'''

  
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
        #should not be zero non-zeros for successful anomalous diagnosis (need atleast 1 zero)
        #every x=count_nonzero in ta which is not 32 i.e., there is a 0(an identified anom) (and so 32-x = non_zero) is a success
        anom_success.append(tf.cast(
            tf.math.not_equal(
                tf.math.reduce_sum(eval_anom[pair_num*32:(pair_num+1)*32]), tf.constant(32, tf.int32)), dtype=tf.int32))
    
    # print("anom_success list:")
    # for t in anom_success:
        # tf.print(t, summarize=-1, end=" ")
        # tf.print("_")
    # print("accumulated_sum:")
    # tf.print(tf.math.accumulate_n(anom_success), summarize=-1)
    # print("divide to get acc:")
    # tf.print(
        # tf.math.divide(
            # tf.cast(tf.math.accumulate_n(anom_success), dtype=tf.int32), 
            # tf.cast(tf.math.divide(tf.cast(num_anom, dtype=tf.int32), tf.constant(32, tf.int32)), dtype=tf.int32)
            # ))
    
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
        
    # print("norm_success list:")
    #for t in norm_success:
        #tf.print(t, summarize=-1, end=" ")
        #tf.print("_")
    #tf.print(tf.math.accumulate_n(norm_success), summarize=-1)
    # print("divide to get acc:")
    
    # tf.print(
        # tf.math.divide(
            # tf.cast(tf.math.accumulate_n(norm_success), dtype=tf.int32), 
            # tf.cast(tf.math.divide(tf.cast(num_norm, dtype=tf.int32), tf.constant(32, tf.int32)), dtype=tf.int32)
            # ))
    
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

"""
TRAIN PROCEDURE
"""
#VERIFIED
def return_seq_model():
    """ 
    Creates the artificial neural network which takes as input a 4096D FC layer of C3D features extracted from a video segment.
    The final layer consists of a single unit which is the anomaly score of a segment.
    The function returns the compiled ANN.
    """

    annMIL = tf.keras.Sequential()
    annMIL.add(tf.keras.layers.InputLayer(input_shape=(4096, ), batch_size=2*BATCH*32)) #we feed in BATCH pairs of anom and norm videos as a 2D tensor (shape: (2*BATCH*32, 4096))
    annMIL.add(tf.keras.layers.Dense(units=512, 
                              input_dim=4096, 
                              activation='relu', 
                              kernel_regularizer=l2(0.001), #acts on weights of network
                              bias_initializer="zeros", 
                              kernel_initializer="glorot_uniform" #waqas uses normal - default on keras is uniform
                              ))
    annMIL.add(tf.keras.layers.Dropout(DROPOUT))
    annMIL.add(tf.keras.layers.Dense(units=32, 
                              kernel_regularizer=l2(0.001), 
                              bias_initializer="zeros", 
                              kernel_initializer="glorot_uniform" #waqas uses normal - default on keras is uniform
                              ))
    annMIL.add(tf.keras.layers.Dropout(DROPOUT))
    annMIL.add(tf.keras.layers.Dense(units=1,
                                   activation="sigmoid",
                                   kernel_regularizer=l2(0.001),
                                   bias_initializer="zeros",
                                   kernel_initializer="glorot_uniform"
                                   ))
    annMIL.summary()
    
    return annMIL

#VERIFIED
def gen_lists(anom_path, norm_path):
    """
    Accumulate list of all paths to training videos - leading up to random selection
    """
    norm_video_paths = []
    anom_video_paths = []
    video_labels = {} 
    for path, collect, l in zip([anom_path, norm_path], [anom_video_paths, norm_video_paths], [1, 0]):
        for video in sorted(os.listdir(path)):
            video_path = os.path.join(path, video)
            collect.append(video_path)
            video_labels[video_path] = l
                      
    # print all paths                 
    # print(reduce(string_util, norm_video_paths))
    # print(reduce(string_util, anom_video_paths))
    return anom_video_paths, norm_video_paths, video_labels

#VERIFIED
def train_model_gen(annMIL):
    """
    This function uses the ann object and path to previously computed C3D features (with labels) to train the network in batches.
    """
     
    meta = {}
    params = {}
    params["num_epoch"] = EPOCH 
    params["batch"] = BATCH
    params["test_batch"] = TEST_BATCH
    params["lambda_2"] = LAMBDA2
    params["lambda_3"] = LAMBDA3
    params["dropout"] = DROPOUT
    params["version"] = VERSION
    params["record_store"] = os.path.join(RECORD_STORE, VERSION)
    params["optimizer"] = str(annMIL.optimizer)
    params["LR"] = LR
    # params["lambda_1"] = LAMBDA1 #EXTENSION
    meta["t_params"] = params
    
   
    anom_video_paths, norm_video_paths, video_labels = gen_lists(ANOM_FEAT, NORM_FEAT)
    shuffle(anom_video_paths) #NB we do not want to feed the data in categories therefore we shuffle video order(esp anom data) 
    shuffle(norm_video_paths) #labels are dicts with keys as video paths so won't lose correspondence with shuffle
    
    # print("anom paths")
    # print(reduce(string_util, anom_video_paths))
    # print("norm paths")
    # print(reduce(string_util, norm_video_paths))
    
    n_folds = N_FOLDS
    anom_chunk = int(len(anom_video_paths)/n_folds)
    norm_chunk = int(len(norm_video_paths)/n_folds) 
    
    results = pd.DataFrame(columns=annMIL.metrics_names)
    kfold_time = 0
    for fold in range(N_FOLDS):
        meta["fold"] = fold+1
        
        print("[FOLD {} of {}]".format(fold+1, n_folds))
        anom_test = anom_video_paths[fold*anom_chunk:(fold+1)*anom_chunk]
        anom_train = anom_video_paths[:fold*anom_chunk] + anom_video_paths[(fold+1)*anom_chunk:]
        norm_test =  norm_video_paths[fold*norm_chunk:(fold+1)*norm_chunk]
        norm_train = norm_video_paths[:fold*norm_chunk] + norm_video_paths[(fold+1)*norm_chunk:]
        print("list lengths: anom test and train, norm test and train")
        print(len(anom_test))
        print(len(anom_train))        
        print(len(norm_test))
        print(len(norm_train)) 
        
        validation_generator = DataGenerator(anom_test, norm_test, video_labels, TEST_BATCH, shuffle=True)
        training_generator = DataGenerator(anom_train, norm_train, video_labels, BATCH, shuffle=True)

        #log metrics
        logger = OutputLogJSON(meta)
        #save model
        lossmodelpath = os.path.join(MODEL_STORE, VERSION, "best_val_loss_model_v{}_fold{}".format(VERSION ,fold+1))
        losscheckpoint = ModelCheckpoint(
                            filepath=lossmodelpath,
                            monitor="val_loss", 
                            verbose=0,
                            save_best_only=True,
                            save_weights_only=True,
                            mode="min"
                            )
        diffmodelpath = os.path.join(MODEL_STORE, VERSION, "best_val_diff_model_v{}_fold{}".format(VERSION, fold+1))
        diffcheckpoint = ModelCheckpoint(
                            filepath=diffmodelpath,
                            monitor="val_difference_of_avg_max",
                            verbose=0,
                            save_best_only=True,
                            save_weights_only=True,
                            mode="max"
                            )
        
        with tf.device("/GPU:0"):
            start = time.time()
            os.system("nvidia-smi")
            annMIL.fit(x=training_generator, 
                       epochs=EPOCH,
                       verbose=0,
                       callbacks=[losscheckpoint, diffcheckpoint, EarlyStopLoss(), logger, OutputCall()],
                       validation_data=validation_generator, 
                        # validation_steps= 1, want to validate on all of validattion set therefore don't give steps - execute till complete
                       validation_freq = FREQ, #patience is 5 validations
                       use_multiprocessing=True,
                       workers=6
                       )
            os.system("nvidia-smi")
            end = time.time()
            elapsed = (end - start)
            kfold_time += elapsed
            print("training the fold took {} seconds".format(end-start))



        print("\n\n\n [FINAL EVALUATIONS]")
        #print("- BEST RATIO MODEL")   # saved at end of early stop    
        #annMIL.load_weights(os.path.join(MODEL_STORE, VERSION, "best_val_loss_ratio_model_v{}_fold{}".format(VERSION, fold+1)))
        #annMIL.evaluate(
        #    x=validation_generator, 
        #    verbose=1,
        #    callbacks=[logger, OutputCall()],
        #)
        print("- BEST VAL LOSS") # saved by model checkpt callback
        annMIL.load_weights(os.path.join(MODEL_STORE, VERSION, "best_val_loss_model_v{}_fold{}".format(VERSION, fold+1)))
        evln = annMIL.evaluate(
            x=validation_generator, 
            verbose=1,
            callbacks=[logger, OutputCall()],
        )
        results.loc[fold] = evln
        results.to_csv(os.path.join(RECORD_STORE, VERSION, "{}-cross-auc-folds.txt".format(VERSION)), sep=" ", float_format='%.4f', header=annMIL.metrics_names)
        
        
        print("- BEST DIFF") # saved by model checkpt callback
        annMIL.load_weights(os.path.join(MODEL_STORE, VERSION, "best_val_diff_model_v{}_fold{}".format(VERSION, fold+1)))
        annMIL.evaluate(
            x=validation_generator, 
            verbose=1,
            callbacks=[logger, OutputCall()],
        )
        
        #ensure best model from this fold is saved by now
        tf.keras.backend.clear_session()
    
    
    fnl = results.describe().fillna("-1")
    print(fnl)
    fnl.to_csv(os.path.join(RECORD_STORE, VERSION, "{}-cross-auc-stats.txt".format(VERSION)), sep=" ", float_format='%.2f', header=annMIL.metrics_names)

    ftimer = open(os.path.join(TIME_STORE, "{}-time.txt".format(VERSION)), "w")
    ftimer.write("training all folds took {} seconds - there are {} folds \n".format(kfold_time, N_FOLDS))
    ftimer.close()
    
    return annMIL

#VERIFIED
def freeze_layers(annMIL, depth):
    """
    freeze from start to 'depth' layers, rest are trainable
    if depth is 1: just freeze weights between input and first hidden layer
    if depth is 2: freeze all except weights between 32 and 1
    if depth is 3: freeze all weights
    """
    for i, layer in enumerate(annMIL.layers):
        print(i, layer.name, layer.trainable)
        
    print("[FREEZE] printing number of layers: expecting 3")
    print(len(annMIL.layers))
    assert len(annMIL.layers) == 3, "not 3 layers - verify freezing method" 
    
    for layer in annMIL.layers:
        layer.trainable = True
    
    for layer in annMIL.layers[:depth]:
        layer.trainable = False
        
    for i, layer in enumerate(annMIL.layers):
        print(i, layer.name, layer.trainable)
        
    adag = tf.keras.optimizers.Adagrad(learning_rate=LR) #waqas

    annMIL.compile(
            optimizer=adag, 
            loss=ranking_loss, 
            metrics=[
                max_anom_scores_metric, 
                max_norm_scores_metric,
                difference_of_avg_max,
                tf_auc_VLE, 
                ]  
            )

    return annMIL
    
#VERIFIED
def main():
    tf.config.run_functions_eagerly(True)

    if os.path.exists(os.path.join(RECORD_STORE, VERSION)) or os.path.exists(os.path.join(MODEL_STORE, VERSION)):
        print("update version!")
        exit()
    else:
        os.mkdir(os.path.join(RECORD_STORE, VERSION))
        os.mkdir(os.path.join(MODEL_STORE, VERSION))
        
    
    print("\n\nSTART OF MAIN:\n")
    gpus = tf.config.list_physical_devices('GPU')   
    cpus = tf.config.list_physical_devices('CPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    logical_cpus = tf.config.experimental.list_logical_devices('CPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    print(len(cpus), "Physical CPUs,", len(logical_cpus), "Logical CPU")

    # if len(os.listdir(MODEL_STORE)) != 0:
            # annMIL = load_model(MODEL_STORE)
    # else:
    annMIL = return_seq_model()
    
    #adadelta = tf.keras.optimizers.Adadelta(learning_rate=LR, epsilon=1e-08) #waqas use
    adag = tf.keras.optimizers.Adagrad(learning_rate=LR) #waqas

    annMIL.compile(
            optimizer=adag, 
            loss=ranking_loss, 
            metrics=[
                max_anom_scores_metric, 
                max_norm_scores_metric,
                difference_of_avg_max,
                tf_auc_VLE, 
                ]  
            )
    
    annMIL.save(os.path.join(MODEL_STORE, "arch"))
    
    if WEIGHT_PATH != None:
        if "waqas" in WEIGHT_PATH:
            annMIL = load_weights_waqas(annMIL, WEIGHT_PATH) #for waqas weight loading
        else:
            annMIL.load_weights(WEIGHT_PATH)
    
    if TRANSFER_LEARN:
        annMIL.trainable = False 
        inputs = tf.keras.Input(shape=(2*BATCH*32, 4096))
        
        x = annMIL(inputs, training=False)
        x = tf.keras.layers.Dense(units=8, 
                              kernel_regularizer=l2(0.001), 
                              bias_initializer="zeros", 
                              kernel_initializer="glorot_uniform" #waqas uses normal - default on keras is uniform
                              )(annMIL.layers[-2].output)
        x = tf.keras.layers.Dropout(DROPOUT)(x)
        outputs = tf.keras.layers.Dense(units=1, 
                        activation="sigmoid",
                        kernel_regularizer=l2(0.001), 
                        bias_initializer="zeros", 
                        kernel_initializer="glorot_uniform" #waqas uses normal - default on keras is uniform
                        )(x)
        
        annMIL2 = tf.keras.Model(inputs, outputs)

        if FREEZE:
            annMIL2 = freeze_layers(annMIL2, 3)

        annMIL2.compile(
        optimizer=adag, 
        loss=ranking_loss, 
        metrics=[
            max_anom_scores_metric, 
            max_norm_scores_metric,
            difference_of_avg_max,
            tf_auc_VLE,
            ]  
        )
        
        annMIL = annMIL2
    
    trained = train_model_gen(annMIL) #input gives c3d features (4096D fc for each segment)    

"""
END TRAIN PROCEDURE
"""    
    
if __name__ == '__main__':
    main()
    







"""
EXPERIMENTAL & DEPRECATED
"""

class AnomVideoLevelEvaluation(tf.keras.metrics.Metric):
    #need to convert this into a metric form as in line above
    #maybe use 1s and 0s with subtract, abs and count_nonzero
    
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(AnomVideoLevelEvaluation, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='AVLE', initializer='zeros')
        self.update_metric = tf.Variable(True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            result = self.compute_anom_video_level_accuracy(y_true, y_pred)
            self.metric_variable.assign_add(result)
            
    def compute_anom_video_level_accuracy(self, y, y_pred):
        y_class = tf.math.round(y_pred)
        anom_preds = y_class[:TEST_BATCH*32]
        num_instances = len(anom_preds)
        eval_anom = tf.math.abs(tf.subtract(anom_preds, y[:TEST_BATCH*32]))
        
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for pair_num in range(TEST_BATCH):
            ta.write(pair_num, tf.cast(tf.math.count_nonzero(eval_anom[pair_num*32:(pair_num+1)*32]), tf.float32)) #should not be zero non-zeros for successful anomalous diagnosis (need atleast 1 zero)
        ta = tf.convert_to_tensor(ta.concat(), dtype=tf.float32)
        success_anom = tf.cast(tf.math.count_nonzero(ta), tf.float32)
        tf.print(success_anom)
        return tf.math.divide(success_anom, num_instances)

    def result(self):
        return self.metric_variable

    def reset_state(self):
        self.metric_variable.assign(0.)
        
class NormVideoLevelEvaluation(tf.keras.metrics.Metric):
    
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(NormVideoLevelEvaluation, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='NVLE', initializer='zeros')
        self.update_metric = tf.Variable(True)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            result = self.compute_norm_video_level_accuracy(y_true, y_pred)
            self.metric_variable.assign_add(result)
            
    def compute_norm_video_level_accuracy(self, y, y_pred):
        y_class = tf.math.round(y_pred)
        norm_preds = y_class[TEST_BATCH*32:TEST_BATCH*32*2]
        num_instances = len(norm_preds)
        eval_norm = tf.math.abs(tf.subtract(norm_preds, y[TEST_BATCH*32:TEST_BATCH*32*2])) 
        tn = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for pair_num in range(TEST_BATCH):
            tn.write(pair_num, tf.cast(tf.math.count_nonzero(eval_norm[pair_num*32:(pair_num+1)*32]), tf.float32)) #shold be 0 non-zeros for successful normal diagnosis (need 32 zeros)
        tn = tf.convert_to_tensor(tn.concat(), dtype=tf.float32)
        success_norm = tf.math.subtract(tf.constant(32, dtype=tf.float32), tf.cast(tf.math.count_nonzero(tn), tf.float32))
        print("success_norm")
        tf.print(success_norm)
        return tf.math.divide(success_norm, num_instances)

    def result(self):
        return self.metric_variable

    def reset_state(self):
        self.metric_variable.assign(0.)
    
class ToggleVLE(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or 
     validation data is run during fit()) toggle metric flag '''
    def on_test_begin(self, logs):
        for metric in self.model.metrics:
            print(metric.name)
            if 'max_anom_scores_metric' in metric.name:
                print("--->toggling max_anom_scores_metric")
                metric.assign(True)
            if 'max_norm_scores_metric' in metric.name:
                print("--->toggling max_norm_scores_metric")
                metric.assign(True)

    def on_test_end(self,  logs):
        for metric in self.model.metrics:
            if 'max_anom_scores_metric' in metric.name:
                metric.assign(False)
            if 'max_norm_scores_metric' in metric.name:
                metric.assign(False)

class customModel(keras.Model):
    def train_step(self, data):
        x, y = data
        print("--->train step")
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # print("y_pred shape is {}".format(y_pred.shape))
            # print("type in y_pred: {}".format(type(y_pred[0])))
            
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) #keras automatically adds reg loss through this

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data): 
        print("test_step")
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
        """
        # x, y = data
        
        # # Compute predictions
        # y_pred = self(x, training=False)
        # y_class = tf.math.round(y_pred)
        # anom_preds = y_class[:BATCH*32]
        # norm_preds = y_class[BATCH*32:BATCH*32*2]
        # eval_anom = tf.math.equal(anom_preds, y[:BATCH*32])
        # eval_norm = tf.math.equal(norm_preds, y[BATCH*32:BATCH*32*2]) 
        
        # ta = tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False)
        # tn = tf.TensorArray(tf.bool, size=0, dynamic_size=True, clear_after_read=False)
        # for pair_num in range(BATCH-1):
        #     tn.write(pair_num, tf.math.reduce_all(eval_norm[pair_num*32:(pair_num+1)*32]))
        #     ta.write(pair_num, tf.math.reduce_any(eval_anom[pair_num*32:(pair_num+1)*32]))
        # ta = tf.convert_to_tensor(ta.concat(), dtype=tf.bool)
        # tn = tf.convert_to_tensor(tn.concat(), dtype=tf.bool)

        # # evaluation = tf.concat(ta, tn)
        # return ta
        """
       
def read_file(path):
    f = open(path, "r")
    lines = f.readlines()
    values = [float(x.strip()) for x in lines]
    return values

def read_video_block(path):
    f= open(path, "r")
    lines = f.readlines()
    data = [l.strip().split(' ') for l in lines]
    data = np.reshape(np.asarray(data).astype("float32"), ((len(data), 4096)))
    return data

def difference(x, y):
    return (x-y)**2
     
def retrieve_features_and_labels_batch(anom_feature_paths, norm_feature_paths, take, test, p):
    """
    This function is given a path to C3D features in the local file system and returns a batch (features from 30 normal & 30 anomalous videos) of features and labels to be used in the
    training process of the model.
    @param normal_path - path to C3D features of normal videos
    @param anomaly_path - path to C3D features of anomaly videos
    @param i - iteration of training data for this i.e after we've split to train and test, we go through train in i blocks of 30 batches (i=24) 
    
    Returns:
    tuple(
    anom_features: list of bags where a bag contains <num_sgm> 4096D vectors
    anom_labels: label 1, per bag
        )
        
    tuple(    
    norm_features: list of bags where a bag contains <num_sgm> 4096D vectors
    norm_labels: label 0, per bag
        )
    """
    # anom_train_count = 810
    # norm_train_count = 800

    anom_features = []
    norm_features = []
    pairs = take
        
    print("len of paths: {} ".format(len(anom_feature_paths)))
    print("len of paths: {} ".format(len(norm_feature_paths)))
        

    indices = np.arange(0, pairs, dtype=int) 

    #collect random set of training paths from all normal and anomalous paths
    for paths, collect in \
        zip([anom_feature_paths, norm_feature_paths], [anom_features, norm_features]):                
        for index in indices:
            if p:
                print("\tselected video:")
                print("\t{}/{}".format(os.path.basename(os.path.dirname(paths[int(index)])), os.path.basename(paths[int(index)])))
            
            bag_of_features = read_video_block(os.path.join(paths[int(index)], "sgm_avg_stack.txt"))
            
            # for sgm in sorted(os.listdir(os.path.join(paths[int(index)], "sgm_avg"))): # preprocess so we have one file with sgm x 4096 vecs for each video
                # sgm_vec_path = os.path.join(paths[int(index)], "sgm_avg", sgm)
                # print(sgm_vec_path)
                # sgm_vec = read_file(sgm_vec_path)
                # bag_of_features.append(sgm_vec)
                # vec_count+=1
                
            #wrap around features till 32
            extra = 32 - bag_of_features.shape[0] 
            while extra != 0:
                # print("EXTRA NEEDED")
                wrap = bag_of_features[:extra]
                bag_of_features = np.vstack((bag_of_features, wrap))
                extra = 32 - bag_of_features.shape[0] 
                
            #print("\t\t\t{} x {} -- bag shape".format(bag_of_features.shape[0], bag_of_features.shape[1]))    

            collect.append(bag_of_features) #32 4096D tensors in 2D tensor form 
    # if p:
        # print("shape of anomaly data:\nfeatures: {}\nlabels: {}".format((np.asarray(anom_features, dtype=object)).shape, np.asarray(anom_labels, dtype=object).shape))
        # print("shape of normal data:\nfeatures: {}\nlabels: {}".format((np.asarray(norm_features, dtype=object)).shape, np.asarray(norm_labels, dtype=object).shape))
        # print("--->END OF BATCH READ")
    # 
        # print("\nRETURNED:\ntuple(\nanom_features: list of bags where a bag contains <num_sgm> 4096D vectors\nanom_labels: label 1, per bag) \
        # \ntuple(\nnorm_features: list of bags where a bag contains <num_sgm> 4096D vectors\nnorm_labels: label 0, per bag)")
        # print("\nBAGS: {} each for anomalous and normal".format(pairs))

    all_data = []        
    for chunk in range(BATCH, pairs+1, BATCH):
        all_data.append(anom_features[chunk-BATCH:chunk])
        all_data.append(norm_features[chunk-BATCH:chunk])
        print("shape of all data: {}".format(np.asarray(all_data).shape))
    
    # anom_data = np.reshape(anom_features, [pairs*32, 4096])
    anom_labels = np.ones((pairs*32, 1)) 
    
    # norm_data = np.reshape(norm_features, [pairs*32, 4096])
    norm_labels = np.zeros((pairs*32, 1)) 
    
    # all_data = np.reshape(np.vstack((anom_data, norm_data)), [pairs*2*32, 4096]) #stack all feature vectors from anom and norm videos
    
    all_data = np.reshape(all_data, [pairs*2*32, 4096]) #stack all feature vectors from anom and norm videos
    all_labels = np.reshape(np.vstack((anom_labels, norm_labels)), [pairs*2*32, 1]) #stack all corresponding labels of above feature vectors
    print("\t\t\t[BAGS]: shape all_data batch: \t\t{}".format(all_data.shape))
    print("\t\t\t[BAGS]: shape all_labels batch: \t{}".format(all_labels.shape))
    return np.hstack((all_data, all_labels))

def train_model(annMIL):
    """
    This function uses the ann object and path to previosuly computed C3D features (with labels) to train the network in batches.
    @param model - ANN object
    """
    
    anom_feature_paths, norm_feature_paths = gen_lists(ANOM_FEAT, NORM_FEAT)
    shuffle(anom_feature_paths) #NB we do not want to feed the data in categories therefore we shuffle video order(esp anom data) 
    shuffle(norm_feature_paths)
    
    # print("anom paths")
    # print(reduce(string_util, anom_feature_paths))
    # print("norm paths")
    # print(reduce(string_util, norm_feature_paths))

    #810 anom train videos
    #800 norm train videos
    
    """
    CALLBACKS
    """
    lossmodelpath = os.path.join(MODEL_STORE, "best_loss_model_epoch{epoch:02d}_loss{val_loss:.2f}_diff{val_difference_of_avg_max:.2f}")
    losscheckpoint = ModelCheckpoint(
        filepath=lossmodelpath, 
        monitor="val_loss",
        verbose=0, 
        save_best_only=True,
        mode="min"
    )
    diffmodelpath = os.path.join(MODEL_STORE, "best_diff_model_epoch{epoch:02d}_loss{val_loss:.2f}_diff{val_difference_of_avg_max:.2f}")
    diffcheckpoint = ModelCheckpoint(
        filepath=diffmodelpath, 
        monitor="val_difference_of_avg_max",
        verbose=0, 
        save_best_only=True,
        mode="max"
    )
    early_stop = EarlyStopLoss()
    
    json_log = open(os.path.join(RECORD_STORE, 'log_v{}.json'.format(VERSION)), mode='wt', buffering=1)
    logger = tf.keras.callbacks.LambdaCallback(
        on_train_end=lambda logs: json_log.close(),
        on_epoch_end = lambda epoch, logs: json_log.write(
            json.dumps({"epoch": epoch,
                "loss": logs["loss"],
                "max_anom_scores_metric": logs["max_anom_scores_metric"],
                "max_norm_scores_metric": logs["max_norm_scores_metric"],
                "compute_AVLE": logs["compute_AVLE"],
                "compute_NVLE": logs["compute_NVLE"],
                "val_loss": logs["val_loss"],
                "val_max_anom_scores_metric": logs["val_max_anom_scores_metric"],
                "val_max_norm_scores_metric": logs["val_max_norm_scores_metric"],
                "val_compute_AVLE": logs["val_compute_AVLE"],
                "val_compute_NVLE": logs["val_compute_NVLE"],
                }) + '\n'
            ) if (epoch % 50 == 0) else False
    )
    output_call = OutputCall()
    """
    END CALLBACKS
    """
    
    n_folds = N_FOLDS
    anom_chunk = int(len(anom_feature_paths)/n_folds)
    norm_chunk = int(len(norm_feature_paths)/n_folds) 
    for fold in range(n_folds):
        print("[FOLD {} of {}]".format(fold+1, n_folds))
        anom_test = anom_feature_paths[fold*anom_chunk:(fold+1)*anom_chunk]
        anom_train = anom_feature_paths[:fold*anom_chunk] + anom_feature_paths[(fold+1)*anom_chunk:]
        norm_test =  norm_feature_paths[fold*norm_chunk:(fold+1)*norm_chunk]
        norm_train = norm_feature_paths[:fold*norm_chunk] + norm_feature_paths[(fold+1)*norm_chunk:]
        # if x:
            #   print("fold {}".format(fold))
            #   print(len(anom_test))
            #   print(len(anom_train))        
            #   print(len(norm_test))
            #   print(len(norm_train)) 
                   
        #if TEST_BACTH and BATCH are same size does not matter if test=False or test=True
        bags_and_labels = retrieve_features_and_labels_batch(anom_train, norm_train, len(norm_train), False, False) #first false is for train, second is for print
        test_bags_and_labels = retrieve_features_and_labels_batch(anom_test, norm_test, len(norm_test), False, False) #true is for test, false is for print
        
        # assert tf.test.is_gpu_available()
        with tf.device("/CPU:0"):
            start = time.time()
            # print("\t\t[ITER WITH EVAL]-------------------------------------------*\n\n\n")
            os.system("nvidia-smi")
            annMIL.fit( x=bags_and_labels[:, :-1], 
                        y=bags_and_labels[:, -1:], 
                        batch_size=2*BATCH*32, 
                        epochs=EPOCH,
                        validation_data=((test_bags_and_labels[:, :-1], test_bags_and_labels[:, -1:])), 
                        shuffle='batch',
                        validation_batch_size=2*TEST_BATCH*32,
                        callbacks=[losscheckpoint, diffcheckpoint, early_stop, logger, output_call]
                        )
            os.system("nvidia-smi")
            # print("\t\t[EVAL COMPLETE]--------------------------------------------*\n\n\n")
            end = time.time()
            print("iteration took {} seconds".format(end-start))
        
        #ensure best model from this fold is saved by now
        os.system("sh cp_store.sh")
        tf.keras.backend.clear_session()
    return annMIL

def return_model():
    """ 
    Creates the artificial neural network which takes as input a 4096D FC layer of C3D features extracted from a video segment.
    The final layer consists of a single unit which is the anomaly score of a segment.
    The function returns the compiled ANN.
    """
    inputs = tf.keras.Input(shape=(4096, ), batch_size=2*BATCH*32) #we feed in BATCH pairs of anom and norm videos as a 2D tensor (shape: (2*BATCH*32, 4096))
    x = tf.keras.layers.Dense(units=512, 
                              input_dim=4096, 
                              activation='relu', 
                              kernel_regularizer=l2(0.001), #acts on weights of network
                              bias_initializer="zeros", 
                              kernel_initializer="glorot_normal" #waqas uses normal - default on keras is uniform
                              )(inputs)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(units=32, 
                              kernel_regularizer=l2(0.001), 
                              bias_initializer="zeros", 
                              kernel_initializer="glorot_normal" #waqas uses normal - default on keras is uniform
                              )(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    output = tf.keras.layers.Dense(units=1,
                                   activation="sigmoid",
                                   kernel_regularizer=l2(0.001),
                                   bias_initializer="zeros",
                                   kernel_initializer="glorot_normal"
                                   )(x)
    annMIL = customModel(inputs, output)
    annMIL.summary()
    return annMIL

def create_ann_model():
    """ 
    Creates the artificial neural network which takes as input a 4096D FC layer of C3D features extracted from a video segmentf.
    The final layer consists of a single unit which is the anomaly score of a segmentf.
    The function returns the compiled ANN.
    """
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=512, input_dim=4096, activation='relu', kernel_regularizer=l2(0.001)))
    ann.add(tf.keras.layers.Dropout(0.6))
    ann.add(tf.keras.layers.Dense(units=32, kernel_regularizer=l2(0.001)))
    ann.add(tf.keras.layers.Dropout(0.6))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=l2(0.001)))
    ann.compile(optimizer='Adagrad', loss='binary_crossentropy', run_eagerly=False) #only eager when debugging - drastically reduces performance
    return ann

#deprecated: use keras funcs
def save_ann_model(model, model_path, weight_path): # Function to save the model
    """
    Once the model has been compiled and trained, this function is called to save the model in JSON form and the the learned weights as a .h5 file.
    The outputted files are to be loaded by test_network and train_network.py for convenient reconstrucion of the network.
    @param model - the model to be saved
    @param model_path - the path where the model is saved in json form
    @param weight_path - the path where the model weights are saved as a matrix
    """
    json_string = model.to_json()
    open(model_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    
    savemat(weight_path, dict)

#deprecated: use keras functions
def load_ann_weights(ann, weight_path): # Function to load the model weights
    """
    This function enables learned parameters to be reproduced in testing by loading them into the model provided as a parameter.
    @param ann - the loaded model which takes on parameters to be loaded into the model
    @param weight_path - path to weights to be loaded into ann
    """
    dict2 = loadmat(weight_path)
    dict = convert_weights(dict2)
    i = 0
    for layer in ann.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return ann

def convert_weights(dict2):
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

def ranking_loss_waq(y_true, y_pred):

    y_true = tf.flatten(y_true)
    y_pred = tf.flatten(y_pred)

    n_seg = 32 # 32 segments per video.
    n_vid = 60 # batch of 30 random normal and 30 random abnormal
    n_exp = n_vid / 2
    num_d = n_seg*n_vid

    sub_max = tf.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = tf.ones_like(y_true) # to sum labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1 = tf.ones_like(y_true)  # for holding the concatenation of summation of scores in the bag.
    sub_l2 = tf.ones_like(y_true) # for holding the concatenation of L2 of score in the bag.

    for ii in range(0, n_vid, 1):
        # collect labels
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        # track abnormal and normal vidoes
        sub_sum_labels = tf.concatenate([sub_sum_labels, tf.stack(tf.sum(mm))])  

        # segment scores
        bag_of_scores = y_pred[ii * n_seg:ii * n_seg + n_seg]
        # keep maximum score of all instances in a bag
        sub_max = tf.concatenate([sub_max, tf.stack(tf.max(bag_of_scores))])         
        # keep sum of scores of all instances in a bag (video)
        sub_sum_l1 = tf.concatenate([sub_sum_l1, tf.stack(tf.sum(bag_of_scores))])   

        z1 = tf.ones_like(bag_of_scores)
        z2 = tf.concatenate([z1, bag_of_scores])
        z3 = tf.concatenate([bag_of_scores, z1])
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = tf.sum(tf.sqr(z))
        sub_l2 = tf.concatenate([sub_l2, tf.stack(z)])

    sub_score = sub_max[num_d:]
    F_labels = sub_sum_labels[num_d:]

    sub_sum_l1 = sub_sum_l1[num_d:] #account for tf.ones_like
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[num_d:]
    sub_l2 = sub_l2[:n_exp]

    indx_nor = tf.tensor.eq(F_labels, 32).nonzero()[0]  # index of normal videos: since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
    indx_abn = tf.tensor.eq(F_labels, 0).nonzero()[0]

    n_norm=n_exp

    sub_norm = sub_score[indx_nor] # Maximum Score for each of abnormal video
    sub_abnorm = sub_score[indx_abn] # Maximum Score for each of normal video

    z = tf.ones_like(y_true)
    for ii in range(0, n_norm, 1):
        sub_z = tf.maximum(1 - sub_abnorm + sub_norm[ii], 0)
        z = tf.concatenate([z, tf.stack(tf.sum(sub_z))])

    z = z[num_d:]  
    z = tf.mean(z, axis=-1) +  0.00008*tf.sum(sub_sum_l1) + 0.00008*tf.sum(sub_l2)  

    return z

def MIL_eqn(max_anom, max_norm):
    return max([0, 1-max_anom+max_norm])
    
"""
END EXPERIMENTAL & DEPRECATED
"""
