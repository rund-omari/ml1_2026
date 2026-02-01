# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 12:53:31 2026

@author: SDK
"""
import streamlit as st 
from sklearn.datasets import load_breast_cancer
import joblib
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
st.title("Breast Cancer Prediction")
#%%
# Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#%%
# Dataset Overview
st.subheader("https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")
st.subheader("Dataset Overview")

col1, col2 = st.columns(2)
col1.metric("Number of Rows", X.shape[0])
col1.metric("Number of Columns", X.shape[1])


class_counts = pd.Series(y).value_counts().rename(
    index={0: "Malignant", 1: "Benign"}
)
col2.write("Samples per Class")
col2.bar_chart(class_counts)

#%%
svm = joblib.load("SVC.pkl")
lr = joblib.load('LR.pkl')
dt = joblib.load('DT.pkl')


svm_model , lr_model , dt_model = st.tabs(["SVM" , "logistic" , "desicion tree"])

for page , model  in [(svm_model , svm) , (lr_model,lr) , (dt_model,dt)] :
    
    with page :
       st.subheader("Enter feature values")
    
       user_input = []
       c=1
       for feature in data.feature_names:
           value = st.number_input(feature, value=0.0,key=f'{model}_{c}')
           user_input.append(value)
           c+=1
    
       if st.button("Predict" ,key = f'{model}'):
           user_input = np.array(user_input).reshape(1, -1) 
    
           prediction = model.predict(user_input)[0]
           if prediction == 0 :
               label = "Malignant"
               st.error(f"Prediction: {label}")
           else :
               label = "Benign"
               st.success(f"Prediction: {label}")
    
           if hasattr(model, "predict_proba"
                      ):
               probs = model.predict_proba(user_input)[0]
               classes = ["Malignant", "Benign"]
    
               fig, ax = plt.subplots()
               ax.pie(probs, labels=classes, autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
               ax.set_title("Class Probabilities")
               st.pyplot(fig)
           else:
               st.warning("This model does not support probability prediction.")
     
    
    
    
    
    
    
    
    
