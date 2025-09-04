# app.py
import streamlit as st
import pandas as pd
from joblib import load

# === Load saved models ===
reg_model, reg_features = load("linear_regression_model.joblib")
clf_model, clf_features = load("rf_classifier.joblib")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict **Final Exam Score** and **Pass/Fail outcome**.")

# === Identify parental education columns ===
parental_cols = [col for col in reg_features if col.startswith("Parental_Education_Level_")]

# === Validation ranges ===
valid_ranges = {
    "Study_Hours_per_Week": (0, 100),
    "Attendance_Rate": (0, 100),
    "Past_Exam_Scores": (0, 100)
}

# === Collect inputs ===
st.subheader("📌 Student Information")
user_input_values = {}

for col in reg_features:
    if col == "Gender":
        user_input_values[col] = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female", horizontal=True)
    elif col == "Internet_Access_at_Home":
        user_input_values[col] = st.radio("Internet Access at Home", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    elif col == "Extracurricular_Activities":
        user_input_values[col] = st.radio("Extracurricular Activities", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    elif col in parental_cols:
        user_input_values[col] = 0  # init one-hot
    else:
        if col in valid_ranges:
            user_input_values[col] = st.slider(col, int(valid_ranges[col][0]), int(valid_ranges[col][1]), int(valid_ranges[col][1])//2)
        else:
            user_input_values[col] = st.number_input(col, value=0.0)

# Parental Education
options = [c.replace("Parental_Education_Level_", "") for c in parental_cols]
chosen_level = st.selectbox("Parental Education Level", options)
one_hot_col_name = f"Parental_Education_Level_{chosen_level}"
if one_hot_col_name in user_input_values:
    user_input_values[one_hot_col_name] = 1.0

# Convert to DataFrame
input_df = pd.DataFrame([user_input_values], columns=reg_features)

# === Prediction ===
if st.button("🔮 Predict Student Performance", use_container_width=True):
    # Predict Score
    predicted_score = reg_model.predict(input_df.values)[0]

    # Predict Pass/Fail
    input_df_class = input_df[clf_features]
    predicted_class = clf_model.predict(input_df_class.values)[0]
    predicted_label = "Pass" if predicted_class == 1 else "Fail"

    # === Display results ===
    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    # Score progress bar
    st.metric("Predicted Final Exam Score", f"{predicted_score:.2f}")
    st.progress(min(int(predicted_score), 100))

    # Pass/Fail outcome
    if predicted_label == "Pass":
        st.success("✅ The student is predicted to **Pass**")
    else:
        st.error("❌ The student is predicted to **Fail**")
