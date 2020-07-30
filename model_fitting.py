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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import load_data

#%% =============================================================================
# set input X and y_true
# =============================================================================
# features, y_trues = load_data.get_first_section()
features, y_trues = load_data.get_20_80_section()

X = features
y = y_trues[:,1]

#remove 0
X=X[y != 0]
y=y[y != 0]

#%% =============================================================================
# modeling
# =============================================================================

# First define the model
# log_reg = LogisticRegression(penalty="none")
log_reg = LogisticRegression(penalty="l2", max_iter=1000)

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
# weights = np.mean(log_reg.coef_, axis=0)
weights = log_reg.coef_

    # The order of importance: ascending
# first: minimum contribution; last: maximum contribution
sorted_index = np.argsort(np.abs(weights))

# recall
recalls = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=3, scoring='recall')
roc = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=3, scoring='roc_auc')

