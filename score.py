# Calculates custom log loss and AUC score by taking mean of all columns

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import sys, os, re, csv, codecs, numpy as np, pandas as pd

def calc_log_loss(y_true, y_pred):
    return np.mean([log_loss(y_true[:, i], y_pred[:, i]) 
                    for i in range(y_true.shape[1])])

def calc_auc_score(y_true, y_pred):
    return np.mean([roc_auc_score(y_true[:, i], y_pred[:, i]) 
                    for i in range(y_true.shape[1])])
    
    