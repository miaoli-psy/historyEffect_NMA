# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:16:17 2020

@author: Miao
"""
# =============================================================================
# import modules
# =============================================================================
from oneibl.onelight import ONE
import numpy as np

# =============================================================================
# ONE object and load data
# =============================================================================
one = ONE()

#all data
eids = one.search(['_ibl_trials.*'])

#get selected probabilities
e_index= []
for i, eid in enumerate(eids):
    p = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
    requiredArray = np.sort(np.unique(p))
    if list(requiredArray) == list(np.array([0.2,0.5,0.8])):
        e_index.append(i)

