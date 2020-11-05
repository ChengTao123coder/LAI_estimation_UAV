# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:54:49 2020

@author: Cheng
"""
import pylab
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics,preprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.externals import joblib

#prepare data
data = pd.read_csv('train.csv',header = None)
x = data.iloc[:80,:12]
y = data.iloc[:80,12]
data2 = pd.read_csv('test.csv',header = None)
test_refle = data2.iloc[:44,:12]
test_lai = data2.iloc[:44,12]

#Step 1：adjust n_estimators
#param_test1 = {
#        'n_estimators': range(10, 100, 5)
#    }
#gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#        learning_rate=0.1, max_depth=5,
#        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#        nthread=4, scale_pos_weight=1, seed=27),
#        param_grid=param_test1, iid=False, cv=5
#     )


#Step 2：Adjust parameter(max_depth and min_child_weight)
#max_depth: The maximum depth of the tree.Increasing this value will make the model more complex and prone to overfitting. A depth of 3-10 is reasonable.
#min_child_weight: The tree building process is stopped if the instance weights in the tree partition are less than the sum defined.

#param_test2 = {
#        'max_depth': range(3, 10, 1),
#        'min_child_weight': range(1, 6, 1),
#    }
#gsearch1 = GridSearchCV(estimator= xgb.XGBRegressor(
#        learning_rate=0.1, n_estimators=55), param_grid=param_test2, cv=5)
#import warnings
#warnings.filterwarnings("ignore")

#Step  3 ：Adjust gamma to reduce the risk of overfitting 
#param_test3 = {
#        'gamma': [0.1*i for i in range(0,10)]
#    }
#gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(
#        learning_rate=0.1, n_estimators=55, max_depth=3, min_child_weight=1), param_grid=param_test3, cv=5)


#Step 4: Adjust the sample sampling mode(subsample and colsample_bytree)
#param_test4 = {'subsample':[ 0.1 * i for i in range(5,9)],
#                      'colsample_bytree':[ 0.1 * i for i in range(6,9)]
#    }
#gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(
#        learning_rate=0.1, n_estimators=55,max_depth=3, min_child_weight=1, gamma=0.9
#        ), param_grid=param_test4, cv=5)

#Step 5: Reduce the learning rate and increase the number of trees
#param_test5 = {'learning_rate':[0.09,0.08,0.07,0.06,0.05,0.04,0.5,0.4,0.3,0.2,0.1,0.05,0.075,0.05,0.04,0.03]
#    }
#gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(
#        n_estimators=55, max_depth=3, min_child_weight=1, gamma=0.9,
#        subsample=0.8,colsample_bytree = 0.6), param_grid=param_test5, cv=5)

#XGB optimal parameter
#gsearch1 = xgb.XGBRegressor(learning_rate = 0.1,
#        n_estimators = 50,
#        max_depth = 3,
#        min_child_weight = 5,
#        gamma = 7.62,
#        colsample_bytree =  0.8,
#        subsample = 0.6,
#        reg_alpha = 0.05,
#        cv=5)

gsearch1.fit(x,y)
y_predict = gsearch1.predict(x)

#print(gsearch1.best_params_)

print('XGBOOST_Modeling_R²=', r2_score(y,y_predict))
print('XGBOOST_Modeling_RMSE=',np.sqrt(metrics.mean_squared_error(y,y_predict)))
print('XGBOOST_Modeling_MAE = ',mean_absolute_error(y,y_predict))
#Save the model
joblib.dump(gsearch1.fit, "VI_XGBoost_gsearch1.pkl")
print('Save the model and training results successfully.')

#Validation
test_lai_predict = gsearch1.predict(test_refle)
print('Validation_R² = ', r2_score(test_lai,test_lai_predict))
print('Validation_RMSE = ',np.sqrt(metrics.mean_squared_error(test_lai,test_lai_predict)))
print('Validation_MAE = ',mean_absolute_error(test_lai,test_lai_predict))

