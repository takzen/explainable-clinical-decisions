import streamlit as st
import pandas as pd
import numpy as np
import shap
from joblib import load
import matplotlib.pyplot as plt

# This line correctly sets the style for dark theme plots.
plt.style.use('dark_background')

# --- Helper Functions ---
@st.cache_resource
def load_assets():
    """Loads the saved model, explainer, and the list of expected columns."""
    model = load('trained_model.joblib')
    explainer = load('shap_explainer.joblib')
    expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    return model, explainer, expected_columns

# --- Main Application Interface ---
st.set_page_config(layout="wide")
st.title("Explainable AI (XAI) for Clinical Decision Support 🩺")
st.write("This app uses a **Random Forest** model to predict heart disease risk and the **SHAP** library to explain why the model made its decision.")

# Load assets with error handling
try:
    model, explainer, expected_columns = load_assets()
except FileNotFoundError:
    st.error("ERROR: Model files not found ('trained_model.joblib', 'shap_explainer.joblib').")
    st.warning("Please make sure you have run the `python model_trainer.py` script first.", icon="⚠")
    st.stop()

# --- Sidebar for Patient Data Input ---
st.sidebar.header("Patient Data")

def user_input_features():
    """Creates interface elements in the sidebar for data input."""
    sex_options = {0: 'Female', 1: 'Male'}
    fbs_options = {0: 'False (<= 120 mg/dl)', 1: 'True (> 120 mg/dl)'}
    exang_options = {0: 'No', 1: 'Yes'}
    
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (chol)', 126, 564, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar (fbs)', options=list(fbs_options.keys()), format_func=lambda x: fbs_options[x])
    restecg = st.sidebar.selectbox('Resting ECG Result (restecg)', [0, 1, 2])
    thalch = st.sidebar.slider('Maximum Heart Rate Achieved (thalch)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', options=list(exang_options.keys()), format_func=lambda x: exang_options[x])
    oldpeak = st.sidebar.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of ST Segment (slope)', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored (ca)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalch': thalch, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    
    features = pd.DataFrame(data, index=[0])
    return features[expected_columns]

input_df = user_input_features()

st.subheader('Processed Patient Input Data')
st.dataframe(input_df)

if st.button('**Predict and Explain**', type="primary"):
    prediction_proba = model.predict_proba(input_df)
    prediction = (prediction_proba[0][1] > 0.5).astype(int)

    st.header('Prediction Result')
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"High Risk of Heart Disease")
        else:
            st.success(f"Low Risk of Heart Disease")
    with col2:
        st.metric(label="Probability of Disease", value=f"{prediction_proba[0][1]:.2%}")

    st.header('Prediction Explanation (AI Insights)')
    
    # --- THIS TEXT BLOCK HAS BEEN RESTORED ---
    st.write("""
    The plots below show which patient features had the greatest impact on the model's decision.
    
    **How to read the Waterfall Plot:**
    - **Red bars (positive values)** = Features that INCREASE the risk of heart disease
    - **Blue bars (negative values)** = Features that DECREASE the risk of heart disease  
    - The plot starts at the baseline risk (E[f(x)]) and shows how each feature pushes the prediction up or down
    - The final value (f(x)) is the model's prediction for this patient
    """)

    # --- FINAL AND GUARANTEED FIX FOR SHAP VISUALIZATION ---
    
    explanation = explainer(input_df)

    explanation_for_class_1 = explanation[0, :, 1]
    
    st.write("### Feature Impact Breakdown")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('#0E1117') 
    ax.set_facecolor('#0E1117')
    shap.plots.waterfall(explanation_for_class_1, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.write("### Feature Importance (Absolute Impact)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.set_facecolor('#0E1117')
    ax2.set_facecolor('#0E1117')
    shap.plots.bar(explanation_for_class_1, show=False)
    st.pyplot(fig2, use_container_width=True)
    plt.close()