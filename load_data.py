# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:27:53 2020

@author: Miao
"""
#%% =============================================================================
# import modules
# =============================================================================
from oneibl.onelight import ONE
import calculate_features
import numpy as np
import copy
# =============================================================================
# ONE object and load data
# =============================================================================
one = ONE()

#all data
eids = one.search(['_ibl_trials.*'])

#get mouse name
name = []
for i, eid in enumerate(eids):
    t = eid.split('/')[0:-1][2]
    name.append(t)

# remove re-occuring names- order preserving
name_no_duplicate = []
[name_no_duplicate.append(i) for i in name if not name_no_duplicate.count(i)]

# name_no_duplicate_set = set(name) #order not preserving

# name_no_duplicate = []
# for n in name_no_duplicate_set:
#     name_no_duplicate.append(n)

#find first data
data_index = []
for n in name_no_duplicate:
    index_t = name.index(n)
    data_index.append(index_t)
    
all_eid = []
for i in data_index:
    eid = eids[i]
    all_eid.append(eid)

#find section that probability are 20 and 80
e_index= []
for i, eid in enumerate(eids):
    p = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
    requiredArray = np.sort(np.unique(p))
    if list(requiredArray) == list(np.array([0.2,0.5,0.8])):
        e_index.append(i)

all_eid_20_80 = []
for i in e_index:
    eid = eids[i]
    all_eid_20_80.append(eid)

new_name = [] #len(new_name) -> 2716
for i, eid in enumerate(all_eid_20_80):
    t = eid.split('/')[0:-1][2]
    new_name.append(t)

new_name_no_duplicate = []
[new_name_no_duplicate.append(i) for i in new_name if not new_name_no_duplicate.count(i)]

data_index_new = [] #len(data_index_new) -> 88
for n in new_name_no_duplicate:
    index_t = new_name.index(n)
    data_index_new.append(index_t)

# #判断每只小鼠在2716个trial中重复几次
# mouse_rep_count = {}
# for i in new_name:
#     if new_name.count(i)>1:
#         mouse_rep_count[i] = new_name.count(i)
# print (mouse_rep_count)

all_eid_new = []
for i in data_index_new:
    eid = eids[i]
    all_eid_new.append(eid)
    
#%% =============================================================================
# features
# =============================================================================

def getF(EID):
    my_features_full = []
    for eid in EID:
        choice = one.load_dataset(eid, '_ibl_trials.choice')
        
        # choice: 1->left; -1->right (change -1 to 2)
        choice = calculate_features.dealwithminu1(choice)
        
        ####f2
        stim_contrast_left = one.load_dataset(eid, '_ibl_trials.contrastLeft')
        
        #以前contrast比0大-》刺激位置在左边
        where_morethan_0 = np.where(stim_contrast_left >= 0)
        stim_contrast_left[where_morethan_0] = 1
        
        #以前是nan》刺激在右面
        posi_is_left_array = np.nan_to_num(stim_contrast_left)
        
        stim_posi_array = copy.copy(posi_is_left_array)
        stim_posi_list = stim_posi_array.tolist()
        
        #change 0 to 2
        stim_posi_list = calculate_features.dealwith0(stim_posi_list)
        
        #change back to array
        stim_posi_array = np.array(stim_posi_list)
        
        ####fb_type: 1->reward; -1->no reward (change -1 to 2)
        fb_type = one.load_dataset(eid, '_ibl_trials.feedbackType')
        fb_type = calculate_features.dealwithminu1(fb_type)
    
        my_features = calculate_features.get_my_feature(choice, posi_is_left_array,stim_posi_array,fb_type)
        my_features_full.append(my_features)
    return my_features_full

# =============================================================================
# t true
# =============================================================================
def getYture(EID):
    my_ytrue_full = []
    for eid in EID:
        #y ture: 1correct or 0incorrect (change 0 to 2)
        y_true = one.load_dataset(eid, '_ibl_trials.feedbackType')
        y_true = calculate_features.dealwithminu1(y_true)
        
        #y true 2: the choice : 1-> left; -1-> right (change to 2 ->right)
        y_true2 = one.load_dataset(eid, '_ibl_trials.choice')
        y_true2 = calculate_features.dealwithminu1(y_true2)
        
        my_ytrue = calculate_features.get_y_ture_array(y_true,y_true2)
        my_ytrue_full.append(my_ytrue)
    return my_ytrue_full
# =============================================================================
# put all feature together
# =============================================================================

def get_first_section():
    my_feature_1section = getF(all_eid)
    my_Ytrue_1section = getYture(all_eid)
    
    my_feature_full = np.concatenate(my_feature_1section, axis=0)
    my_ytrue_full   = np.concatenate(my_Ytrue_1section, axis=0)
    return my_feature_full, my_ytrue_full

def get_20_80_section():
    my_feature_20_80_section = getF(all_eid_new)
    my_Ytrue_20_80_section = getYture(all_eid_new)
    
    my_feature_full_20_80 = np.concatenate(my_feature_20_80_section, axis=0)
    my_ytrue_full_20_80   = np.concatenate(my_Ytrue_20_80_section, axis=0)
    return my_feature_full_20_80, my_ytrue_full_20_80

# features, y_trues = get_first_section()
# features, y_trues = get_20_80_section()
