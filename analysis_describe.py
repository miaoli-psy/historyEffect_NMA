# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:03:18 2020

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

# =============================================================================
# load data
# =============================================================================

# create ONE object
one = ONE()

# search terms
one.search_terms()

# list all dataset
eids = one.search(['_ibl_trials.*'])

# try one session
eid = eids[-1]

# see all data set types
dset_types = one.list(eid)

# load a single dateset
choice = one.load_dataset(eid, dset_types[0])

# load entire object
trials = one.load_object(eid, "_ibl_trials")
for key, value in trials.items():
    print(key, value.shape)
    


#%% =============================================================================
#sample plot -visualization 
# =============================================================================
out = []
for sgn, contrast in ((-1, trials.contrastRight), (+1, trials.contrastLeft)):
    for c in np.unique(contrast)[::sgn]:
        if not np.isnan(c) and (c != 0 or sgn == +1):
            out.append((sgn * c, (trials.choice[contrast == c] == +1).mean()))
out = np.array(out) * 100

plt.figure(figsize=(10, 6))
plt.plot(out[:, 0], out[:, 1], 'o-', lw=4, ms=10)
plt.xlabel("Signed contrast (%)", fontsize = 15)
plt.ylabel("Rightward choice (%)", fontsize = 15)
plt.ylim(0, 100)
plt.title("Psychometric curve for %s" % eid, fontsize = 15);



