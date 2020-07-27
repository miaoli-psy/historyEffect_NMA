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

#f1
choice = one.load_dataset(eid, dset_types[0])

# choice: 1->left; -1->right (change -1 to 2)
for i, c in enumerate(choice):
    if c == -1:
        choice[i] = 2

#f2
stim_contrast_left = one.load_dataset(eid, dset_types[1])

#f3
fd_type = one.load_dataset(eid, dset_types[3])

#y ture: correct or incorrect
y_true = one.load_dataset(eid, dset_types[3])

#y true 2: the choice : 1-> left; -1-> right (change to 2 ->right)
y_true2 = one.load_dataset(eid, dset_types[0])

# load entire object
# trials = one.load_object(eid, "_ibl_trials")
# for key, value in trials.items():
#     print(key, value.shape)
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
    for i in range(10, len(raw_choice)):
        res_count = count_num(raw_choice[i-10:i])
        count_feature.append(res_count)
    return count_feature


# [0,1,2,3,4,5,6,7,8,9,10]
# index == 10: [index:0, index:9]
def get_past_trials_choice(raw_choice):
    '''
    calcualte features that the choice of 
    the past 1-10 choice
    '''
    features_choice = [[], [], [],[],[],[],[],[],[],[]]
    
    for i in range(10, len(raw_choice)):
        for i, num in enumerate(raw_choice[i-10:i]):
            features_choice[i].append(num)
    return features_choice


f1 = cal_feature_choice(choice)
f1_array = np.array(f1)

# cal past choices
f1_pastchoice = get_past_trials_choice(choice)

#past choice to correct format n_trials * features
f1_pastchoice =[[row[i] for row in f1_pastchoice] for i in range(len(f1_pastchoice[0]))]

#to array
f1_pastchoice = np.array(f1_pastchoice)


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
    for i in range(10, len(posi_is_left_array)):
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
    for i in range(10, len(raw_fbtype)):
        res_count = count_num(raw_fbtype[i-9:i+1])
        count_feature.append(res_count)
    return count_feature

f3_array = cal_feature_fb(fd_type)
f3_array = np.array(f3_array)


def cal_feature_stimu_contrast(raw_contrastLeft):
    pass


def get_my_feature():
    
    f1 = cal_feature_choice(choice)
    f1_array = np.array(f1)
    
    f2_array = cal_feature_stim_posi(stim_contrast_left)
    f2_array = np.array(f2_array)

    f3_array = cal_feature_fb(fd_type)
    f3_array = np.array(f3_array)

    my_features = np.column_stack((f1_array,f2_array,f3_array))
    
    return my_features

# my_features = get_my_feature()

#%% =============================================================================
# true values --> y
# =============================================================================

# calculate true values: eg: the respone of current trial is correct or not (0/1)

def get_y_ture(y_true, y_true2):
    for i, res in enumerate(y_true):
        if res == -1:
            y_true[i] = 0
    for i, y in enumerate(y_true2):
        if y == -1:
            y_true2[i] = 2
    # y_true2 = y_true2
    
    # y_true = y_true[9:]
    return y_true[10:], y_true2[10:]

my_y_true, my_y_true2 = get_y_ture(y_true, y_true2)

