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
import calculate_features
import load_data


features, y_trues = load_data.get_all()

X = features
y = y_trues[:,1]

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

accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=3) # k=8 crossvalidation

# Get the weight of each feature
# In other word, the importance of the features in prediction of y
weights = np.mean(log_reg.coef_, axis=0)

# The order of importance: ascending
# first: minimum contribution; last: maximum contribution
sorted_index = np.argsort(np.abs(weights))
