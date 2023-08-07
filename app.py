import streamlit as st
from Predict import show_predict_page
from EDA import show_explore_page



# Sidebar - Specify parameter settings
page = st.sidebar.selectbox("Explore or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()

