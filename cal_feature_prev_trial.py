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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# create ONE object
one = ONE()

# list all dataset
eids = one.search(['_ibl_trials.*'])
# 316, 333, 347, ,366, 369
# try one session
eid = eids[316]


# e_index= []
# for i, eid in enumerate(eids):
#     p = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
#     requiredArray = np.sort(np.unique(p))
#     if list(requiredArray) == list(np.array([0.2,0.5,0.8])):
#         e_index.append(i)


# see all data set types
dset_types = one.list(eid)

# load a single dateset
obj = one.load_object(eid, "_ibl_trials")
# for key, value in obj.items():
#     print(key)
#     print(value)
#     print()
### calculate feature 
choice = one.load_dataset(eid, '_ibl_trials.choice')
choice[np.where(choice==-1)]=0
contrast_left = one.load_dataset(eid, 'contrastLeft')
contrast_left = np.nan_to_num(contrast_left)
contrast_right = one.load_dataset(eid, 'contrastRight')
contrast_right = np.nan_to_num(contrast_right)
contrast = contrast_right-contrast_left
# plt.scatter(contrast,contrast)
reward = one.load_dataset(eid, 'feedbackType')
p_left = one.load_dataset(eid, 'probabilityLeft')
# plt.scatter(p_left,p_left)
# plt.plot(p_left)


pos=np.where(p_left==0.5)
pos=pos[0][-1]+1

choice=choice[pos:]
contrast=contrast[pos:]
reward=reward[pos:]


trial_interval=np.arange(101)
trial_interval=trial_interval[1:]
# define compute_accuracy function to calculate accuracy
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




# # calculate training accuracy based on previous 

# x1=choice[:-1]
# x2=contrast[:-1]
# x3=reward[:-1]
# # x4=p_left[:-1]

# X=np.stack((x1, x2, x3))
# X=X.T

# # X=np.random.permutation(X)

# y=choice[1:]

# # First define the model
# log_reg = LogisticRegression(penalty="none")

# #Then fit it to data
# log_reg.fit(X, y)

# y_pred = log_reg.predict(X)


# train_accuracy = compute_accuracy(X, y, log_reg)
# print(f"Accuracy on the training data: {train_accuracy:.2%}")

# accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=8) # k=8 crossvalidation

# weight= log_reg.coef_

# X.shape



train_accu=[]
cv_accu=np.empty([101, 10])
cv_weight=np.empty([101, 3])
# calculate training accuracy based on previous 
for i in trial_interval:
    x1=choice[:-i]
    x2=contrast[:-i]
    x3=reward[:-i]
    # x4=p_left[:-1]
    
    X=np.stack((x1, x2, x3))
    X=X.T
    
    # X=np.random.permutation(X)
    
    y=choice[i:]
    
    # First define the model
    log_reg = LogisticRegression(penalty="none")
    
    #Then fit it to data
    log_reg.fit(X, y)
    
    y_pred = log_reg.predict(X)
    
    
    train_accuracy = compute_accuracy(X, y, log_reg)
    train_accu.append(train_accuracy)
    print(f"Accuracy on the training data: {train_accuracy:.2%}")
    accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=10) # k=10 crossvalidation
    weight= log_reg.coef_
    cv_accu[i,:]=accuracies
    cv_weight[i,:]=weight



X.shape

plt.plot(train_accu)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.xlabel('Trial distance')
plt.ylabel('Accuracy')
plt.title('Training accuracy in 100 models')
plt.savefig('Accu.svg',  dpi=1200)
plt.savefig('Accu.png',  dpi=1200)


plt.plot(cv_accu[1:,:])
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.xlabel('Trial distance')
plt.ylabel('Accuracy')
plt.title('10-fold CV accuracy in 100 models')
plt.savefig('CV.svg',  dpi=1200)
plt.savefig('CV.png',  dpi=1200)



import seaborn as sns
ax = sns.heatmap(cv_weight[1:,:],cmap="YlGnBu")
# plt.show()
plt.xlabel('Features')
plt.ylabel('Trial distance')
plt.title('Weights in 100 models')
plt.savefig('heatmap.svg',  dpi=1200)
plt.savefig('heatmap.png',  dpi=1200)

# plt.plot(cv_weight[1:,0])

# import seaborn as sns
# ax = sns.heatmap(cv_accu.T[:,1:],cmap="YlGnBu")
# plt.show()



# ## prediction
# # 316, 333, 347, ,366, 369
# # try one session
# eid = eids[333]

# # see all data set types
# dset_types = one.list(eid)

# # load a single dateset
# obj = one.load_object(eid, "_ibl_trials")
# # for key, value in obj.items():
# #     print(key)
# #     print(value)
# #     print()
# ####f1
# choice = one.load_dataset(eid, '_ibl_trials.choice')
# choice[np.where(choice==-1)]=0
# contrast_left = one.load_dataset(eid, 'contrast_left')
# contrast_left = np.nan_to_num(contrast_left)
# contrast_right = one.load_dataset(eid, 'contrast_right')
# contrast_right = np.nan_to_num(contrast_right)
# contrast = contrast_right-contrast_left
# # plt.scatter(contrast,contrast)
# reward = one.load_dataset(eid, 'feedbackType')
# p_left = one.load_dataset(eid, 'probabilityLeft')
# # plt.scatter(p_left,p_left)
# plt.plot(p_left)


# x1=choice[:-1]
# x2=contrast[:-1]
# x3=reward[:-1]
# # x4=p_left[:-1]

# X=np.stack((x1, x2, x3))
# X=X.T

# y=choice[1:]


# y_pred = log_reg.predict(X)

# ### 

# def compute_accuracy(X, y, model):
#   """Compute accuracy of classifier predictions.
  
#   Args:
#     X (2D array): Data matrix
#     y (1D array): Label vector
#     model (sklearn estimator): Classifier with trained weights.
#   Returns:
#     accuracy (float): Proportion of correct predictions.  
#   """
#   y_pred = model.predict(X)
#   accuracy = (y == y_pred).mean()

#   return accuracy

# train_accuracy = compute_accuracy(X, y, log_reg)
# print(f"Accuracy on the test data: {train_accuracy:.2%}")

# accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=8) # k=8 crossvalidation

# weight= log_reg.coef_

# X.shape


