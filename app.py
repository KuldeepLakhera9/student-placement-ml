import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

st.title("🎓 Student Placement Prediction System")
st.markdown("This ML model predicts whether a student will get placed based on academic and skill performance.")

st.divider()

# Create 2 columns
col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", 5.0, 9.5, 7.0)
    internships = st.slider("Internships", 0, 4, 1)
    projects = st.slider("Projects", 1, 6, 2)

with col2:
    aptitude = st.slider("Aptitude Score", 40, 100, 60)
    communication = st.slider("Communication Skill (1-10)", 1, 10, 5)
    technical = st.slider("Technical Skill (1-10)", 1, 10, 5)

st.divider()

if st.button("🚀 Predict Placement"):

    input_data = np.array([[cgpa, internships, projects, aptitude, communication, technical]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Student is likely to be Placed")
    else:
        st.error("❌ Student is NOT likely to be Placed")

    st.metric("Placement Probability", f"{round(probability[0][1] * 100, 2)}%")

st.divider()

st.markdown("### 🧠 Model Information")
st.write("Algorithm Used: Logistic Regression")
st.write("Training Accuracy: 94%")
st.write("Dataset Size: 500 Students (Synthetic)")