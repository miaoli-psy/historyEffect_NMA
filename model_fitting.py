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
import time
import seaborn as sns
import pandas as pd
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
y_pred_prob = log_reg.predict_proba(X)

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

accuracies = cross_val_score(LogisticRegression(penalty='none'), X, y, cv=5) # k=8 crossvalidation

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

#%% =============================================================================
# permuatation
# =============================================================================
start = time.time()
permu_results = []
for perm in range(5):
    X=np.random.permutation(X)
    log_reg.fit(X, y)
    y_pred = log_reg.predict(X)
    perm_accuracy=round(compute_accuracy(X, y, log_reg),6)
    permu_results.append(perm_accuracy)
    
end = time.time()
print((end - start)/60)
DF = pd.DataFrame()

#%%=============================================================================
# plot
# =============================================================================

#put plot data in to dataframe
DF = pd.DataFrame()
# get first coloum
df_x = np.concatenate((permu_results, accuracies),axis = None)
#get secton coloum
df_y_1 = ['permu'] *5
df_y_2 = ['cross_val'] *5
df_y = df_y_1 + df_y_2

DF['values']= df_x
DF['type']= df_y

# SaLat1 = DF.pivot_table(values = 'SaLat', index = 'subject_nr', columns = ['Pro_con', 'Sal_con','Sa2Tar'], aggfunc = 'mean')
# to_plot =SaLat1.melt(value_name = 'SaLat')

ax=sns.barplot(x = 'values', y = 'type', data = DF,capsize=.07,ci=68)

sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.barplot(x="day", y="total_bill", data=tips)
    
sns.despine(top=True, right=True, left=False, bottom=False,trim = False,offset=10)
