import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load trained model
# -------------------------------
model_data = pickle.load(open("model/model.pkl", "rb"))

best_model = model_data["best_model"]
best_model_name = model_data["best_model_name"]

# -------------------------------
# Title
# -------------------------------
st.title("🎓 Student Placement Prediction System")

st.write(
"""
This system predicts whether a student is likely to get placed 
based on academic performance and skill metrics using 
Machine Learning.
"""
)

st.divider()

# -------------------------------
# Example Button (for demo)
# -------------------------------
if st.button("Use Example Student Data"):
    cgpa = 8.0
    internships = 2
    projects = 3
    aptitude = 75
    communication = 7
    technical = 8
else:
    cgpa = st.slider("CGPA", 5.0, 9.5, 7.0)
    internships = st.slider("Number of Internships", 0, 4, 1)
    projects = st.slider("Number of Projects", 1, 6, 2)
    aptitude = st.slider("Aptitude Score", 40, 100, 60)
    communication = st.slider("Communication Skill (1-10)", 1, 10, 5)
    technical = st.slider("Technical Skill (1-10)", 1, 10, 5)

st.divider()

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Placement"):

    input_data = np.array([[cgpa, internships, projects,
                            aptitude, communication, technical]])

    prediction = best_model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ The student is likely to be Placed")
    else:
        st.error("❌ The student is Not Likely to be Placed")

        st.subheader("Suggestions to Improve Placement Chances")

        if cgpa < 7:
            st.write("• Improve academic performance (increase CGPA).")

        if technical < 6:
            st.write("• Improve technical skills like programming and problem solving.")

        if internships < 1:
            st.write("• Try doing internships for practical experience.")

        if communication < 6:
            st.write("• Improve communication and presentation skills.")

st.divider()

# -------------------------------
# Model Information
# -------------------------------
st.subheader("Model Used")

st.write(f"The prediction is generated using **{best_model_name}** machine learning model.")

