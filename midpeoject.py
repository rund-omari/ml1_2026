import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 

st.title("Breast Cancer Prediction")

#%%
# Load Data
data = sklearn.datasets.load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#%%
# Dataset Overview
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
# Model Selection
model_name = st.selectbox(
    "Choose Model",
    ["logistic", "svc", "dt"]
)

model = joblib.load(f"{model_name}_model.pkl")

#%%
# Prediction Section
st.subheader("Enter feature values")

user_input = []
for feature in data.feature_names:
    value = st.number_input(feature, value=0.0)
    user_input.append(value)

if st.button("Predict"):
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
