import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Load history & model
hist = pd.read_csv("student_history.csv")
model = joblib.load("student_model.pkl")

st.title("🎓 Roll Number Based Student Performance Predictor")

# ----------------------------- INPUT SECTION -----------------------------

roll_no = st.number_input("Enter Roll Number", min_value=101, max_value=150)

# Fetch student history
history = hist[hist["roll_no"] == roll_no]

if history.empty:
    st.error("❌ No historical data found for this roll number.")
    st.stop()

st.subheader("📌 Student Previous History")
st.dataframe(history)

# Extract LAST year's final grade as previous_grade
last_prev_grade = history.sort_values("year").iloc[-1]["final_grade"]

st.info(f"Last Recorded Previous Grade: **{last_prev_grade}**")

# ----------------------------- CURRENT INPUT -----------------------------

st.header("📝 Enter Current Year Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.slider("Study Hours", 1.0, 12.0, 6.0)

with col2:
    attendance = st.slider("Attendance (%)", 40, 100, 80)

with col3:
    sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, 7.0)

assignments = st.slider("Assignments Completed", 1, 10, 7)
# age = st.slider("Age", 15, 20, 17)

# Prepare model input
student_input = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_grade": last_prev_grade,
    "sleep_hours": sleep_hours,
    "assignments_completed": assignments,
    
}])

# ----------------------------- PREDICTION -----------------------------

if st.button("Predict Performance"):
    pred_label = model.predict(student_input)[0]

    # ---------- COLOR-CODED PERFORMANCE RESULT ----------
    if pred_label == 'High':
        st.success(f"Performance Level (ML): {pred_label}")
    elif pred_label == 'Medium':
        st.warning(f"Performance Level (ML): {pred_label}")
    else:
        st.markdown(
            f"<div style='padding:12px; border-radius:8px; background-color:#ff4d4d; color:white; font-weight:bold;'>"
            f"Performance Level (ML): {pred_label}</div>",
            unsafe_allow_html=True
        )
    # -----------------------------------------------------

    # Weighted score calculation
    weighted_score = (
        study_hours / 12 * 20 +
        attendance / 100 * 25 +
        last_prev_grade / 100 * 35 +
        sleep_hours / 10 * 10 +
        assignments / 10 * 10
    )

    st.metric("Estimated Grade Score", f"{int(weighted_score)}/100")
