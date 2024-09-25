import streamlit as st
import pandas as pd
from pickle import load
import os

prediction = 0

model = load(open("../models/model_xgbregressor_42_studentperformance.sav", "rb"))

st.title("Student Performance")

val1 = st.slider("Attendance", min_value=60.0, max_value=100.0, step=1)
val2 = st.slider("Hours studied", min_value=1.0, max_value=60.0, step=1)
val3 = st.slider("Access to Resources", min_value=0.0, max_value=2.0, step=1)
val4 = st.slider("Parental Involvement", min_value=0.0, max_value=2.0, step=1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    pred_class = prediction
    st.write("Prediction:", pred_class)