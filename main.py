import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# Define paths - these should match where you saved your model and metadata
MODEL_PATH = "stroke_prediction_pipeline.joblib"
METADATA_PATH = "stroke_prediction_metadata.joblib"

st.set_page_config(layout="wide")
st.title("Stroke Prediction App")

st.write("Loading model and metadata...")

try:
    loaded_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    st.success("Model and metadata loaded successfully!")
except Exception as e:
    st.error("Could not load the model or metadata. Please ensure the paths are correct and files exist.")
    st.exception(e)
    st.stop()

st.write("Please enter the patient's information to predict their stroke risk.")

# Get feature lists from metadata
numerical_features = metadata.get("numerical_features", [])
categorical_features = metadata.get("categorical_features", [])

input_data = {}

# Create input fields for numerical features
st.header("Numerical Features")
num_cols = st.columns(len(numerical_features))
for i, feature in enumerate(numerical_features):
    with num_cols[i]:
        if feature == "age":
            input_data[feature] = st.slider("Age", 0.0, 100.0, 45.0, step=0.1)
        elif feature == "avg_glucose_level":
            input_data[feature] = st.slider("Average Glucose Level", 50.0, 300.0, 100.0, step=0.1)
        elif feature == "bmi":
            # BMI has missing values in original data, impute during preprocessing in pipeline
            input_data[feature] = st.slider("BMI", 10.0, 60.0, 25.0, step=0.1)

# Create input fields for categorical features
st.header("Categorical Features")
# Split into two rows for better layout if many features
cat_cols1 = st.columns(len(categorical_features) // 2 + len(categorical_features) % 2)
cat_cols2 = st.columns(len(categorical_features) // 2)

for i, feature in enumerate(categorical_features):
    if i < len(categorical_features) // 2 + len(categorical_features) % 2:
        with cat_cols1[i]:
            pass
    else:
        with cat_cols2[i - (len(categorical_features) // 2 + len(categorical_features) % 2)]:
            pass

    if feature == "gender":
        input_data[feature] = st.selectbox("Gender", ["Male", "Female", "Other"])
    elif feature == "hypertension":
        input_data[feature] = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    elif feature == "heart_disease":
        input_data[feature] = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    elif feature == "ever_married":
        input_data[feature] = st.selectbox("Ever Married", ["Yes", "No"])
    elif feature == "work_type":
        input_data[feature] = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    elif feature == "Residence_type":
        input_data[feature] = st.selectbox("Residence Type", ["Urban", "Rural"])
    elif feature == "smoking_status":
        input_data[feature] = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])


if st.button("Predict Stroke Risk"):
    try:
        # Create a DataFrame from the input data
        # Ensure the order of columns matches the original training data's X
        # The pipeline's preprocessor will handle one-hot encoding and scaling
        input_df = pd.DataFrame([input_data])

        # Predict probability
        probability = loaded_model.predict_proba(input_df)[0, 1]

        # Get classification threshold from metadata or use default 0.5
        classification_threshold = metadata.get("classification_threshold", 0.5)
        prediction = int(probability >= classification_threshold)

        st.subheader("Prediction Results:")
        st.write(f"Predicted probability of stroke: **{probability:.2%}**")
        if prediction == 1:
            st.error("**High risk of stroke.**")
        else:
            st.success("**Low risk of stroke.**")

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)
