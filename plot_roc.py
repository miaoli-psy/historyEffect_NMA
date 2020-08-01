# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:48:56 2020

@author: XIAOWAN
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  
import model_fitting

# y_01 = np.where(model_fitting.y==2, 0, model_fitting.y) 
# y_pred_01 = np.where(model_fitting.y_pred==2, 0, model_fitting.y_pred)

# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(model_fitting.y, 
                              model_fitting.y_pred_prob[:,1], pos_label=2)
roc_auc = auc(fpr,tpr) 
 

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('ROC curve', fontsize = 15)
plt.legend(loc="lower right")
plt.show()