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
choice = one.load_dataset(eid, dset_types[0])
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
    
    arg*:
    return*
    feature_posi_num:
    
    '''
    pass


def cal_feature_stimu_contrast(raw_contrastLeft):
    pass

def cal_feature_fb(raw_fbtype):
    pass


#%% =============================================================================
# true values --> y
# =============================================================================

# calculate true values: eg: the respone of current trial is correct or not (0/1)
for i, res in enumerate(y_true):
    if res == -1:
        y_true[i] = 0

y_true = y_true[9:]