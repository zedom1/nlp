# -*- coding:utf8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split

xgb1 = xgb.sklearn.XGBClassifier(learning_rate =0.1,n_estimators=50,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8, colsample_bytree=0.8, objective='multi:softmax', num_class= 5, scale_pos_weight=1, seed=27)
train = pd.read_csv("../data/xgboost.train.csv")
y = train.label
print(y)
X = train.drop(['question_id', 'label'], axis=1)


parameters = {'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}


clf = GridSearchCV(xgb1, parameters, n_jobs=-1)
clf.fit(X, y)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)
with open('cv_result.csv','w') as f:
    cv_result.to_csv(f)
    
print('The parameters of the best model are: ')
print(clf.best_params_)

y_pred = clf.predict(X)
print(classification_report(y_true=y, y_pred=y_pred))