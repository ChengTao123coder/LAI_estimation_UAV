# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:38:27 2020

@author: Cheng
"""

import pylab
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import metrics,preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.externals import joblib

##prepare data
data = pd.read_csv('train.csv',header = None)
x = data.iloc[:80,:12]
y = data.iloc[:80,12]
data2 = pd.read_csv('test.csv',header = None)
test_refle = data2.iloc[:44,:12]
test_lai = data2.iloc[:44,12]

#Modeling
model = SVR(kernel='rbf')
param_grid = {'C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001,0.01,0.1,10,20]}
grid_search = GridSearchCV(SVR(), param_grid, n_jobs = -1, verbose=1)
grid_search.fit(x, y)
best_parameters = grid_search.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)
scores =grid_search.cv_results_['mean_test_score']
#.reshape(9,9)
model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
#model = SVR(kernel='rbf',C=best_parameters['C'], cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=best_parameters['gamma'], max_iter=-1, shrinking=True, tol=0.001, verbose=False)

model.fit(x,y)

y_predict = model.predict(x)

print('Modeling_R²=', r2_score(y,y_predict))
print('Modeling_RMSE = ',np.sqrt(np.sum(y-y_predict)**2/len(y)))
print('Modeling_MAE = ',mean_absolute_error(y,y_predict))
#Save the model
joblib.dump(model.fit, "SVR_model.pkl")
print('Save the model and training results successfully.')

#Validation

test_lai_predict = model.predict(test_refle)
print('Validation_R² = ', r2_score(test_lai.values,test_lai_predict))
print('Validation_RMSE = ',np.sqrt(metrics.mean_squared_error(test_lai,test_lai_predict)))
print('Validation_MAE = ',mean_absolute_error(test_lai,test_lai_predict))
print('test_lai_predict = ',test_lai_predict)

