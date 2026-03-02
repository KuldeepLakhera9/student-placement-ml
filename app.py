import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------
# Page Config
# -----------------------------------------

st.set_page_config(page_title="Placement Analytics Pro", page_icon="🎓")
st.title("🎓 AI-Based Placement Analytics Dashboard")
st.markdown("Multi-Model Comparison + Live Prediction System")

st.divider()

# -----------------------------------------
# Load Model Data
# -----------------------------------------

model_data = pickle.load(open("model/model.pkl", "rb"))

best_model = model_data["best_model"]
best_model_name = model_data["best_model_name"]
all_accuracies = model_data["all_accuracies"]
conf_matrices = model_data["confusion_matrices"]
features = model_data["features"]

# -----------------------------------------
# Sidebar Info
# -----------------------------------------

st.sidebar.header("📊 Model Summary")
st.sidebar.write(f"🏆 Best Model: {best_model_name}")

st.sidebar.subheader("All Model Accuracies")

for model_name, acc in all_accuracies.items():
    st.sidebar.write(f"{model_name}: {round(acc*100,2)}%")

st.divider()

# -----------------------------------------
# Accuracy Comparison Chart
# -----------------------------------------

st.subheader("📈 Model Accuracy Comparison")

fig_acc, ax_acc = plt.subplots()
ax_acc.bar(all_accuracies.keys(), 
           [acc*100 for acc in all_accuracies.values()])
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.set_xticklabels(all_accuracies.keys(), rotation=20)

st.pyplot(fig_acc)

st.divider()

# -----------------------------------------
# User Input Section
# -----------------------------------------

st.subheader("🔮 Predict Placement")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", 5.0, 9.5, 7.0)
    internships = st.slider("Internships", 0, 4, 1)
    projects = st.slider("Projects", 1, 6, 2)

with col2:
    aptitude = st.slider("Aptitude Score", 40, 100, 60)
    communication = st.slider("Communication Skill", 1, 10, 5)
    technical = st.slider("Technical Skill", 1, 10, 5)

if st.button("🚀 Predict Using Best Model"):

    input_data = np.array([[cgpa, internships, projects,
                            aptitude, communication, technical]])

    prediction = best_model.predict(input_data)

    # Check if model supports probability
    if hasattr(best_model, "predict_proba"):
        probability = best_model.predict_proba(input_data)
        placement_prob = probability[0][1] * 100
    else:
        placement_prob = None

    st.subheader("📌 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Likely to be Placed")
    else:
        st.error("❌ Not Likely to be Placed")

    if placement_prob is not None:
        st.metric("Placement Probability", f"{round(placement_prob, 2)}%")

st.divider()

# -----------------------------------------
# Confusion Matrix Viewer
# -----------------------------------------

st.subheader("📊 Confusion Matrix Viewer")

selected_model = st.selectbox(
    "Select Model to View Confusion Matrix",
    list(conf_matrices.keys())
)

cm = conf_matrices[selected_model]

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")

st.pyplot(fig_cm)