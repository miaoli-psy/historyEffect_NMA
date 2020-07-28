# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:02:32 2020

@author: Miao
"""
#%% =============================================================================
# import modules
# =============================================================================
# from oneibl.onelight import ONE
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn
# from ibllib.misc import pprint
# import copy

#%% =============================================================================
# useful funtions
# =============================================================================

def swap_matrx(input_2d_list):
    '''
    行变列，列变行
    '''
    return [[row[i] for row in input_2d_list] for i in range(len(input_2d_list[0]))]

def dealwithminu1(input_list):
    '''
    将input是-1改成2
    '''
    output_list = []
    for v in input_list:
        if v == -1:
            v = 2
            output_list.append(v)
        else:
            output_list.append(v)
    return output_list

def dealwith0(input_list):
    '''
    将input是0改成2
    '''
    output_list = []
    for v in input_list:
        if v == 0:
            v = 2
            output_list.append(v)
        else:
            output_list.append(v)
    return output_list

# [1,1,1,1,1,0,0,0,0,0]
# [0, 5, 5]
def count_num(curr_nums:list) -> list:
    '''
    计算list中value-1，0,1的个数
    '''
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

#%% =============================================================================
# calculate features
# =============================================================================

def count_n_in_past_trials(raw_input):
    '''
    calcualte the number of features (e.g. choice, stim posi, fb) in the past 10 trais
    
    arg: raw imput list or array with size (n_trials,)
    
    return: number of features in past 10 trials
    '''
    # raw_choice = raw_choice.tolist()
    count_feature = []
    # count_feature = np.zeros(len(raw_choice)-9)
    for i in range(10, len(raw_input)):
        res_count = count_num(raw_input[i-10:i])
        count_feature.append(res_count)
    return count_feature

# [0,1,2,3,4,5,6,7,8,9,10]
# index == 10: [index:0, index:9]
def get_past_features(raw_choice):
    '''
    calcualte features that the choice of 
    the past 1-10 choice
    '''
    features_choice = [[], [], [],[],[],[],[],[],[],[]]
    
    for i in range(10, len(raw_choice)):
        for i, num in enumerate(raw_choice[i-10:i]):
            features_choice[i].append(num)
    return features_choice

#%% =============================================================================
# get features
# =============================================================================

def get_my_feature(choice, posi_is_left_array,stim_posi_array, fb_type):
    '''
    f1_array: 前十个trial里小鼠选左的次数
    f1_pastchoice： 前十个trial中，第n个trial小鼠的选择； left(1), right(2)
    
    f2_array: 前十个trial里刺激呈现在左的次数
    f2_pastposi： 前十个trial中，第n个trial刺激呈现的位置； left(1), right(2)
    
    f3_array： 前十个trial里获得奖励的次数
    f3_pastfb: 前十个trial中第n个trial小鼠获得奖励情况；reward(1), whitenoise(2)
    '''
    
    #### feature1: mice choice
    f1 = count_n_in_past_trials(choice)
    f1_array = np.array(f1)
    
    # cal past choices
    f1_pastchoice = get_past_features(choice)
    
    #past choice to correct format n_trials * features
    f1_pastchoice = swap_matrx(f1_pastchoice)
    
    #to array
    f1_pastchoice = np.array(f1_pastchoice)
    
    #### feature2: stim posi
    f2 = count_n_in_past_trials(posi_is_left_array)
    f2_array = np.array(f2)
    
    # cal past stim posi
    f2_pastposi =get_past_features(stim_posi_array)
    
    #to correct format
    f2_pastposi = swap_matrx(f2_pastposi)
    
    #to array
    f2_pastposi = np.array(f2_pastposi)
    
    #### feature3: feadback
    f3 = count_n_in_past_trials(fb_type)
    f3_array = np.array(f3)
    
    #cal past fb
    f3_pastfb = get_past_features(fb_type)
    
    # to correct format
    f3_pastfb = swap_matrx(f3_pastfb)
    
    # to array
    f3_pastfb = np.array(f3_pastfb)
    
    
    my_features = np.column_stack((f1_array,f1_pastchoice, f2_array, f2_pastposi, f3_array, f3_pastfb))
    
    return my_features

# my_features = get_my_feature()

# %%=============================================================================
# calculate y_ture
# =============================================================================
def get_y_ture_array(y_true, y_true2):
    '''
    get y_true
    
    arg: raw y_true
    
    return: y_true array that can put into models
    '''
    return np.array(list(zip(y_true[10:],y_true2[10:])))
