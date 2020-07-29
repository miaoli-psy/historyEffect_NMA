#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 19:49:51 2020

@author: juhuang
"""

from oneibl.onelight import ONE
import numpy as np
import matplotlib.pyplot as plt
# import seaborn
# from ibllib.misc import pprint
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# create ONE object
one = ONE()

# list all dataset
eids = one.search(['_ibl_trials.*'])
# 316, 333, 347, ,366, 369
# try one session
eid = eids[316]

# see all data set types
dset_types = one.list(eid)

# load a single dateset
obj = one.load_object(eid, "_ibl_trials")
# for key, value in obj.items():
#     print(key)
#     print(value)
#     print()
####f1
choice = one.load_dataset(eid, dset_types[0])
choice[np.where(choice==-1)]=0
contrast_left = one.load_dataset(eid, dset_types[1])
contrast_left = np.nan_to_num(contrast_left)
contrast_right = one.load_dataset(eid, dset_types[2])
contrast_right = np.nan_to_num(contrast_right)
contrast = contrast_right-contrast_left
# plt.scatter(contrast,contrast)
reward = one.load_dataset(eid, dset_types[3])
p_left = one.load_dataset(eid, dset_types[9])
# plt.scatter(p_left,p_left)
plt.plot(p_left)


x1=choice[:-1]
x2=contrast[:-1]
x3=reward[:-1]
# x4=p_left[:-1]

X=np.stack((x1, x2, x3))
X=X.T

y=choice[1:]





# X=f1_array.reshape(-1, 1) 
# X=X.T
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

weight= log_reg.coef_

X.shape


## prediction
# 316, 333, 347, ,366, 369
# try one session
eid = eids[333]

# see all data set types
dset_types = one.list(eid)

# load a single dateset
obj = one.load_object(eid, "_ibl_trials")
# for key, value in obj.items():
#     print(key)
#     print(value)
#     print()
####f1
choice = one.load_dataset(eid, dset_types[0])
choice[np.where(choice==-1)]=0
contrast_left = one.load_dataset(eid, dset_types[1])
contrast_left = np.nan_to_num(contrast_left)
contrast_right = one.load_dataset(eid, dset_types[2])
contrast_right = np.nan_to_num(contrast_right)
contrast = contrast_right-contrast_left
# plt.scatter(contrast,contrast)
reward = one.load_dataset(eid, dset_types[3])
p_left = one.load_dataset(eid, dset_types[9])
# plt.scatter(p_left,p_left)
plt.plot(p_left)


x1=choice[:-1]
x2=contrast[:-1]
x3=reward[:-1]
# x4=p_left[:-1]

X=np.stack((x1, x2, x3))
X=X.T

y=choice[1:]


y_pred = log_reg.predict(X)

### 

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
print(f"Accuracy on the test data: {train_accuracy:.2%}")

accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=8) # k=8 crossvalidation

weight= log_reg.coef_

X.shape


