import streamlit as st
import pickle
import numpy as np
import pandas as pd


filename = 'best_catboost_model.pkl'

with open(filename, 'rb') as file:
    model = pickle.load(file)


st.title("🏠 Energy Efficiency Prediction App")
st.write("Enter building features to predict **Energy Efficiency (Heating & Cooling Load)**")


relative_compactness = st.slider('Relative Compactness', 0.6, 1.0, 0.8)
surface_area = st.number_input('Surface Area', min_value=0.0, value=600.0)
wall_area = st.number_input('Wall Area', min_value=0.0, value=300.0)
roof_area = st.number_input('Roof Area', min_value=0.0, value=150.0)
overall_height = st.selectbox('Overall Height', [3.5, 7.0])
orientation = st.selectbox('Orientation', [2, 3, 4, 5])
glazing_area = st.slider('Glazing Area', 0.0, 0.5, 0.25)
glazing_area_dist = st.selectbox('Glazing Area Distribution', [0, 1, 2, 3, 4, 5])


if st.button("🔮 Predict Energy Efficiency"):
    input_df = pd.DataFrame([{
        'Relative_Compactness': relative_compactness,
        'Surface_Area': surface_area,
        'Wall_Area': wall_area,
        'Roof_Area': roof_area,
        'Overall_Height': overall_height,
        'Orientation': orientation,
        'Glazing_Area': glazing_area,
        'Glazing_Area_Distribution': glazing_area_dist
    }])

    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Energy Efficiency: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

