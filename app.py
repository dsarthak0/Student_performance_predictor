# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import json
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Predictor (Roll-based)", layout="wide")

# Load data and model
HIST_PATH = "student_history.csv"
MODEL_PATH = "rf_grade_model.pkl"
FEATURES_JSON = "feature_names.json"

@st.cache_data
def load_history(path=HIST_PATH):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

hist_df = load_history()
model = load_model()

with open(FEATURES_JSON, "r") as f:
    feature_names = json.load(f)

st.title("🎓 Roll Number Based Student Performance Predictor (Random Forest)")

# Input roll number
roll_no = st.number_input("Enter Roll Number", min_value=int(hist_df["roll_no"].min()), max_value=int(hist_df["roll_no"].max()), value=int(hist_df["roll_no"].min()))

# fetch history
history = hist_df[hist_df["roll_no"] == roll_no].sort_values("year")
if history.empty:
    st.error("No historical data found for this roll number.")
    st.stop()

st.subheader("📌 Student Historical Records")
st.dataframe(history.reset_index(drop=True))

# compute historical aggregates using ALL available previous records
def compute_hist_features(history_df):
    # use all rows in history_df
    hist_mean_final = history_df["final_grade"].mean()
    hist_last_final = history_df["final_grade"].iloc[-1]
    if len(history_df) >= 2:
        slope = np.polyfit(history_df["year"].values, history_df["final_grade"].values, 1)[0]
        std_final = history_df["final_grade"].std(ddof=0)
    else:
        slope = 0.0
        std_final = 0.0
    hist_mean_study = history_df["study_hours"].mean()
    hist_mean_attendance = history_df["attendance"].mean()
    hist_count = len(history_df)
    return {
        "hist_mean_final": float(hist_mean_final),
        "hist_last_final": float(hist_last_final),
        "hist_trend": float(slope),
        "hist_std_final": float(std_final),
        "hist_mean_study": float(hist_mean_study),
        "hist_mean_attendance": float(hist_mean_attendance),
        "hist_count": int(hist_count)
    }

hist_feats = compute_hist_features(history)

st.markdown("**Historical aggregated features (computed from all available years):**")
st.json(hist_feats)

st.header("📝 Enter Current Inputs (for prediction)")
col1, col2, col3 = st.columns(3)
with col1:
    study_hours = st.slider("Study Hours (today/avg)", 0.0, 12.0, float(round(hist_feats["hist_mean_study"] if not np.isnan(hist_feats["hist_mean_study"]) else 6.0,1)), 0.1)
with col2:
    attendance = st.slider("Attendance (%)", 40, 100, int(round(hist_feats["hist_mean_attendance"] if not np.isnan(hist_feats["hist_mean_attendance"]) else 80)))
with col3:
    sleep_hours = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.1)

assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 80)
parental_support = st.selectbox("Parental Support (1-5)", [1,2,3,4,5], index=2)
extracurricular = st.selectbox("Extracurricular (0-5)", [0,1,2,3,4,5], index=2)

# Build the model input vector - ensure same order as feature_names
model_input = {
    "hist_mean_final": hist_feats["hist_mean_final"],
    "hist_last_final": hist_feats["hist_last_final"],
    "hist_trend": hist_feats["hist_trend"],
    "hist_std_final": hist_feats["hist_std_final"],
    "hist_mean_study": hist_feats["hist_mean_study"],
    "hist_mean_attendance": hist_feats["hist_mean_attendance"],
    "hist_count": hist_feats["hist_count"],
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_grade": hist_feats["hist_last_final"],  # we use last recorded final as previous_grade
    "parental_support": parental_support,
    "extracurricular": extracurricular,
    "sleep_hours": sleep_hours,
    "assignments_completed": assignments_completed
}

input_df = pd.DataFrame([model_input])[feature_names]  # ensure column order

# Predict
if st.button("Predict Future Grade"):
    pred_grade = float(model.predict(input_df)[0])
    pred_grade = max(0.0, min(100.0, pred_grade))  # clamp

    # Map to performance
    if pred_grade >= 80:
        perf = "High"
    elif pred_grade >= 60:
        perf = "Medium"
    else:
        perf = "Low"

    # Show color-coded performance with Low in red
    if perf == "High":
        st.success(f"Performance (predicted): {perf}")
    elif perf == "Medium":
        st.warning(f"Performance (predicted): {perf}")
    else:
        st.markdown(
            f"<div style='padding:12px; border-radius:8px; background-color:#ff4d4d; color:white; font-weight:bold;'>"
            f"Performance (predicted): {perf} (Predicted Grade: {pred_grade:.1f}%)</div>",
            unsafe_allow_html=True
        )

    st.metric("Predicted Final Grade (%)", f"{pred_grade:.1f}")

    # Compare with historical average & last
    colA, colB = st.columns(2)
    with colA:
        st.write(f"Historical mean final grade: **{hist_feats['hist_mean_final']:.1f}%**")
        st.write(f"Historical last final grade: **{hist_feats['hist_last_final']:.1f}%**")
    with colB:
        if pred_grade > hist_feats["hist_mean_final"]:
            st.success("Predicted grade is above the student's historical mean 👍")
        else:
            st.info("Predicted grade is at or below the student's historical mean.")

    # Graphs
    st.divider()
    st.header("📈 Historical Trend and Comparison")

    # Historical trend line
    fig1 = px.line(history, x="year", y="final_grade", markers=True, title=f"Roll {roll_no} - Final Grade by Year")
    fig1.add_scatter(x=[history["year"].max()+0.5], y=[pred_grade], mode="markers+text", text=["Predicted"], textposition="top center")
    st.plotly_chart(fig1, use_container_width=True)

    # Bar: previous vs predicted vs historical mean
    comp_df = pd.DataFrame({
        "label": ["Historical Last", "Historical Mean", "Predicted"],
        "grade": [hist_feats["hist_last_final"], hist_feats["hist_mean_final"], pred_grade]
    })
    fig2 = px.bar(comp_df, x="label", y="grade", title="Previous vs Mean vs Predicted", range_y=[0,100])
    st.plotly_chart(fig2, use_container_width=True)

    # Feature contribution snapshot (raw current inputs)
    st.header("⚙️ Current Input Snapshot")
    fi_df = pd.DataFrame({
        "feature": ["study_hours","attendance","sleep_hours","assignments_completed","parental_support","extracurricular"],
        "value": [study_hours, attendance, sleep_hours, assignments_completed, parental_support, extracurricular]
    })
    fig3 = px.bar(fi_df, x="feature", y="value", title="Current Inputs")
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Model predicts final grade using a Random Forest trained on per-roll historical aggregates + current inputs.")
