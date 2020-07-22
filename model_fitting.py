#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:38:26 2020

@author: juhuang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:02:32 2020
@author: Miao
"""
#%% =============================================================================
# import modules
# =============================================================================
from oneibl.onelight import ONE
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from ibllib.misc import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#%%=============================================================================

# load data
# =============================================================================

# create ONE object
one = ONE()

# list all dataset
eids = one.search(['_ibl_trials.*'])

# try one session
eid = eids[10]

# see all data set types
dset_types = one.list(eid)

# load a single dateset

#f1
choice = one.load_dataset(eid, dset_types[0])

#f2
stim_contrast_left = one.load_dataset(eid, dset_types[1])

#f3
fd_type = one.load_dataset(eid, dset_types[3])

#y ture
y_true = one.load_dataset(eid, dset_types[3])

# load entire object
trials = one.load_object(eid, "_ibl_trials")
for key, value in trials.items():
    print(key, value.shape)
#%% =============================================================================
# intrested features
# =============================================================================

# [1,1,1,1,1,0,0,0,0,0]
# [0, 5, 5]
def count_num(curr_nums:list) -> list:
    assert len(curr_nums) == 10
    count_minus_one =0 
    count_zero = 0
    count_one = 0
    for num in curr_nums:
        if num == -1:
            count_minus_one += 1
        elif num == 0:
            count_zero += 1
        elif num == 1:
            count_one += 1
    # return [count_minus_one, count_zero, count_one]
    return count_one

# temp_list = [1]*100
# temp_list.extend([0]*100)
# temp_list.extend([-1]*100)
# for i in range(9, len(temp_list)):
#     res_count = count_num(temp_list[i-9:i+1])
#     print(res_count)
# # calculate features


def cal_feature_choice(raw_choice):
    '''
    calcualte the number of choice 1 from 
    past 10 trials
    
    arg*
    raw_choice: array of int, eg (1009,)
    
    return
    feature_choice: array of int, eg (1009-9,)
    '''
    # raw_choice = raw_choice.tolist()
    count_feature = []
    # count_feature = np.zeros(len(raw_choice)-9)
    for i in range(9, len(raw_choice)):
        res_count = count_num(raw_choice[i-9:i+1])
        count_feature.append(res_count)
    return count_feature

f1 = cal_feature_choice(choice)
f1_array = np.array(f1)


def cal_feature_stim_posi(raw_contrastLeft):
    '''
    calculate the number of stimulus that 
    appeared on the left visual field
    
    '''
    # find 0: 以前的contrast是0
    where_0 = np.where(stim_contrast_left == 0)
    
    #把0改成-99
    stim_contrast_left[where_0] = -99
    
    #以前contrast比0大-》刺激位置在左边
    where_morethan_0 = np.where(stim_contrast_left > 0)
    stim_contrast_left[where_morethan_0] = 1
    
    #以前是nan》刺激在右面
    posi_is_left_array = np.nan_to_num(stim_contrast_left)
    
    count_feature = []
    for i in range(9, len(posi_is_left_array)):
        res_count = count_num(posi_is_left_array[i-9: i+1])
        count_feature.append(res_count)

    return count_feature

f2_array = cal_feature_stim_posi(stim_contrast_left)
f2_array = np.array(f2_array)

def cal_feature_fb(raw_fbtype):
    '''
    calculate the number of reward in the 
    previous 10 trials
    '''
    count_feature = []
    # count_feature = np.zeros(len(raw_choice)-9)
    for i in range(9, len(raw_fbtype)):
        res_count = count_num(raw_fbtype[i-9:i+1])
        count_feature.append(res_count)
    return count_feature

f3_array = cal_feature_fb(fd_type)
f3_array = np.array(f3_array)


def cal_feature_stimu_contrast(raw_contrastLeft):
    pass

#%% =============================================================================
# true values --> y
# =============================================================================

# calculate true values: eg: the respone of current trial is correct or not (0/1)
for i, res in enumerate(y_true):
    if res == -1:
        y_true[i] = 0

y_true = y_true[9:]


# y = data["choices"]
# X = data["spikes"]
# X=f1_array.reshape(-1, 1) 
X=np.stack((f1_array, f2_array, f3_array))
X=X.T
y=y_true
# First define the model
log_reg = LogisticRegression(penalty="none")

#Then fit it to data
log_reg.fit(X, y)

y_pred = log_reg.predict(X)

def compute_accuracy(X, y, model):
  """Compute accuracy of classifier predictions.
  
  Args:
    X (2D array): Data matrix
    y (1D array): Label vector
    model (sklearn estimator): Classifier with trained weights.
  Returns:
    accuracy (float): Proportion of correct predictions.  
  """
  y_pred = model.predict(X)
  accuracy = (y == y_pred).mean()

  return accuracy

train_accuracy = compute_accuracy(X, y, log_reg)
print(f"Accuracy on the training data: {train_accuracy:.2%}")

accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=8) # k=8 crossvalidation

X.shape
