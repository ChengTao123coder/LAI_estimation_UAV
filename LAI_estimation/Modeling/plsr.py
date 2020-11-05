# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:58:04 2020

@author: Cheng
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:58:09 2020

@author: Cheng
"""


import pylab
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression

#prepare data

data = pd.read_csv('train.csv',header = None)
x = data.iloc[:80,:12]
y = data.iloc[:80,12]
data2 = pd.read_csv('test.csv',header = None)
test_refle = data2.iloc[:44,:12]
test_lai = data2.iloc[:44,12]

#Modeling

model = PLSRegression(copy=True, max_iter=500, n_components=2, scale=True,
        tol=1e-06)
model.fit(x,y)

y_predict = model.predict(x)

print('Modeling_R²=', r2_score(y,y_predict))
print('Modeling_RMSE=',np.sqrt(metrics.mean_squared_error(y,y_predict)))
print('Modeling_MAE = ',mean_absolute_error(y,y_predict))

#Save the model
joblib.dump(model.fit, "PLSR_model.pkl")
print('Save the model and training results successfully.')
#Validation

test_lai_predict = model.predict(test_refle)
MSE=np.sum(np.power((test_lai.values - test_lai_predict),2))/len(test_lai.values)
R2=1-MSE/np.var(test_lai.values)
print('Validation_R² = ', r2_score(test_lai,test_lai_predict))
print('Validation_RMSE = ',np.sqrt(metrics.mean_squared_error(test_lai,test_lai_predict)))
print('Validation_MAE = ',mean_absolute_error(test_lai,test_lai_predict))
print('test_lai_predict = ',test_lai_predict)

 
