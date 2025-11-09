import streamlit as st
import joblib 
import numpy as np
import pandas as pd 
import os 

filename = 'best_catboost_model.joblib' 

if not os.path.exists(filename):
    st.error(f"FATAL ERROR: Model file '{filename}' not found in the repository.")
    st.error("Please ensure 'best_catboost_model.joblib' is in the root of your GitHub repo.")
    st.stop()

try:
    model = joblib.load(filename)
except Exception as e:
    st.error(f"Error loading model: {e}.")
    st.error("This is likely due to a version mismatch. Ensure 'catboost==1.2.8' is in requirements.txt and the model was saved with the same version.")
    st.stop()


st.title("🏠 Energy Efficiency Prediction App")
st.write("Enter building features to predict **Energy Efficiency (Heating & Cooling Load)**")
st.markdown("---")


relative_compactness = st.slider('Relative Compactness', 0.6, 1.0, 0.8)
surface_area = st.number_input('Surface Area', value=600.0)
wall_area = st.number_input('Wall Area', value=300.0)
roof_area = st.number_input('Roof Area', value=150.0)
overall_height = st.selectbox('Overall Height', [3.5, 7.0])
orientation = st.selectbox('Orientation', [2, 3, 4, 5], help="Direction of the main facade.")
glazing_area = st.slider('Glazing Area (Ratio)', 0.0, 0.5, 0.25, help="Area of windows relative to floor area.")
glazing_area_dist = st.selectbox('Glazing Area Distribution', [0, 1, 2, 3, 4, 5], help="Distribution factor of the glazing area.")
st.markdown("---")

if st.button("🔮 Predict Energy Efficiency", type="primary"):
    input_data = {
        'Relative_Compactness': relative_compactness,
        'Surface_Area': surface_area,
        'Wall_Area': wall_area,
        'Roof_Area': roof_area,
        'Overall_Height': overall_height,
        'Orientation': orientation,
        'Glazing_Area': glazing_area,
        'Glazing_Area_Distribution': glazing_area_dist
    }
    
    input_df = pd.DataFrame([input_data])
    
    
    input_df = input_df.astype({
        'Relative_Compactness': 'float64', 
        'Surface_Area': 'float64', 
        'Wall_Area': 'float64', 
        'Roof_Area': 'float64',
        'Overall_Height': 'float64', 
        'Orientation': 'int64', 
        'Glazing_Area': 'float64',
        'Glazing_Area_Distribution': 'int64' 
    })

    try:
        prediction_result = model.predict(input_df)[0]
        st.success("✅ Prediction Successful!")
        
        if isinstance(prediction_result, np.ndarray) and prediction_result.shape == (2,):
            heating_load = prediction_result[0]
            cooling_load = prediction_result[1]
            st.metric(label="🌡️ Predicted Heating Load", value=f"{heating_load:.2f}")
            st.metric(label="❄️ Predicted Cooling Load", value=f"{cooling_load:.2f}")
        elif isinstance(prediction_result, (float, int, np.floating)):
             st.metric(label="📊 Predicted Energy Load", value=f"{prediction_result:.2f}")
        else:
             st.info(f"Prediction made, but result format is unusual: {prediction_result}")

    except Exception as e:
        if "CatBoostError" in str(e):
             st.error("⚠️ Prediction Failed: CatBoost couldn't match input features. Check feature names and types.")
        else:
             st.error(f"⚠️ An unknown error occurred during prediction: {e}")