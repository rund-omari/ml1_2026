# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 12:52:23 2026

@author: SDK
"""

#%%
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


#%%
#load data 
data = load_breast_cancer()
X = pd.DataFrame(data.data , columns=data.feature_names )
y = data.target

#%%
#split data 
x_train , x_test , y_train , y_test = train_test_split(X,y ,
                                                       test_size = 0.2 ,
                                                       random_state=42,stratify=y)


#%%
#pipeline
models = {
    "LR" : Pipeline([
        ("scaler" , StandardScaler()),
        ("model" , LogisticRegression())
        ]),
    
    "SVC" :Pipeline([
        ("scaler" , StandardScaler()),
        ("model" , SVC(probability=True))
        ]),
    
    "DT" :Pipeline([
        ("model" , DecisionTreeClassifier(max_depth=6))
        ]),
    
    }

for name , model in models.items() :
    print(f"\n===== {name.upper()} =====")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print(classification_report(y_test, y_pred))

    joblib.dump(model ,f"{name}.pkl" )
