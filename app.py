# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

# Page config
st.set_page_config(page_title="Student Performance Predictor (4-year)", page_icon="🎓", layout="wide")

# Paths
MODEL_PATH = 'student_performance_model.pkl'
FEATURES_JSON = 'feature_names.json'
HISTORICAL_CSV = 'data/students_2021_2024_combined.csv'

# Utility: load model + feature names
@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_JSON, 'r') as f:
        feature_names = json.load(f)
    return model, feature_names

@st.cache_data
def load_historical(path=HISTORICAL_CSV):
    df = pd.read_csv(path)
    return df

# Load resources (fail early with clear message)
try:
    model, feature_names = load_model_and_features()
except Exception as e:
    st.error(f"Failed to load model/features. Run train_model.py first. Error: {e}")
    st.stop()

try:
    hist_df = load_historical()
except Exception as e:
    st.error(f"Failed to load historical dataset. Run data_generator.py first. Error: {e}")
    st.stop()

# Header
st.title("🎓 Student Performance Predictor — ML + 4-Year Historical Comparison")
st.markdown("This app uses a Random Forest trained on synthetic 4-year student data, and compares your input to historical distributions.")

st.divider()

# Sidebar model info
with st.sidebar:
    st.header("📊 Model info")
    st.info("Random Forest classifier trained on synthetic 4-year dataset (2021-2024).")
    st.write("Features used:")
    st.write(feature_names)
    st.markdown("---")
    st.header("Filter historical data")
    years = sorted(hist_df['year'].unique().tolist())
    selected_years = st.multiselect("Which years to include in historical comparison", years, default=years)
    st.markdown("Use these toggles to control the historical baseline used in comparisons.")

# Main layout
col1, col2 = st.columns([2,1])

with col1:
    st.header("📝 Enter Student Data")
    c1, c2 = st.columns(2)
    with c1:
        study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 5.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 85, 1)
        previous_grade = st.slider("Previous Grade (%)", 0, 100, 75, 1)
        parental_support = st.slider("Parental Support (1–5)", 1, 5, 3, 1)
    with c2:
        extracurricular = st.slider("Extracurricular Activities (0–5)", 0, 5, 2, 1)
        sleep_hours = st.slider("Sleep Hours per Day", 4.0, 12.0, 7.0, 0.5)
        assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 90, 1)
        age = st.slider("Age", 15, 20, 17, 1)

    predict_button = st.button("🚀 Predict & Compare")

with col2:
    st.header("🎯 Quick Historical Stats")
    # Show simple historical aggregated stats for selected years
    hist_subset = hist_df[hist_df['year'].isin(selected_years)]
    if len(hist_subset) == 0:
        st.warning("No historical data for the selected years.")
    else:
        st.metric("Records (selected years)", len(hist_subset))
        st.write("Median study hours:", round(hist_subset['study_hours'].median(),1))
        st.write("Median attendance:", round(hist_subset['attendance'].median(),1))
        st.write("Median final grade:", round(hist_subset['final_grade'].median(),1))

# Prediction + comparison logic
if predict_button:
    # Prepare input row
    student = {
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_grade': previous_grade,
        'parental_support': parental_support,
        'extracurricular': extracurricular,
        'sleep_hours': sleep_hours,
        'assignments_completed': assignments_completed,
        'age': age
    }

    input_df = pd.DataFrame([student])[feature_names]

    # ML prediction
    pred_label = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    class_labels = model.classes_
    proba_dict = {class_labels[i]: round(proba[i]*100,2) for i in range(len(class_labels))}
    confidence = max(proba) * 100

    # Historical comparisons (selected years only)
    h = hist_subset.copy()
    # compute percentiles of the historical distribution for each feature
    percentiles = {}
    for feat in feature_names:
        # compute percentile rank of student[feat] within historical data
        pct = (h[feat] <= student[feat]).mean() * 100
        percentiles[feat] = round(pct, 1)

    # compute a weighted "expected grade" using same weighting as dataset generation
    weighted_score = (
        student['study_hours'] / 12 * 25 +
        student['attendance'] / 100 * 20 +
        student['previous_grade'] / 100 * 20 +
        student['parental_support'] / 5 * 15 +
        student['extracurricular'] / 5 * 10 +
        student['sleep_hours'] / 12 * 5 +
        student['assignments_completed'] / 100 * 5
    )
    predicted_grade_calc = min(100, max(0, weighted_score * 0.7 + student['previous_grade'] * 0.3))
    predicted_grade_calc = round(predicted_grade_calc, 1)

    # Combine ML confidence and historical percentile into a final score (simple weighted fusion)
    # Note: This is a heuristic combination you can tune.
    # We convert ML predicted class into a numeric baseline: Low=0, Medium=50, High=100 (midpoints)
    class_to_score = {'Low': 30, 'Medium': 60, 'High': 85}
    ml_score = class_to_score.get(pred_label, 50)
    # historical percentile average (higher means better)
    hist_percentile_avg = np.mean([percentiles['study_hours'], percentiles['attendance'], percentiles['previous_grade'],
                                  percentiles['parental_support'], percentiles['extracurricular'],
                                  percentiles['sleep_hours'], percentiles['assignments_completed']])
    # final_score weighted: 60% ML, 40% historical trend
    final_score = round((0.6 * ml_score + 0.4 * hist_percentile_avg), 1)

    # Display results
    st.divider()
    st.header("🔎 Prediction Result")

    # Left: ML result
    colA, colB = st.columns([1,1])
    with colA:
        if pred_label == 'High':
            st.success(f"Performance Level (ML): {pred_label}")
        elif pred_label == 'Medium':
            st.warning(f"Performance Level (ML): {pred_label}")
        else:
            st.error(f"Performance Level (ML): {pred_label}")
        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.write("Class probabilities:")
        prob_df = pd.DataFrame(list(proba_dict.items()), columns=['Level','Probability (%)'])
        st.table(prob_df.set_index('Level'))

    with colB:
        st.metric("Predicted Grade (formula)", f"{predicted_grade_calc}%")
        st.metric("Final Combined Score", f"{final_score}/100")
        st.write("Interpretation:")
        if final_score >= 75:
            st.success("Likely to perform well — keep it up! 🎉")
        elif final_score >= 50:
            st.info("Average performance — there's room to improve.")
        else:
            st.warning("At-risk — consider interventions (study plan / assignments).")

    # Historical percentile display: show as bar chart
    st.divider()
    st.header("📊 How you compare to historical students (selected years)")

    pct_df = pd.DataFrame({
        'Feature': ['Study Hours','Attendance','Previous Grade','Parental Support','Extracurricular','Sleep Hours','Assignments Completed'],
        'Percentile': [
            percentiles['study_hours'], percentiles['attendance'], percentiles['previous_grade'],
            percentiles['parental_support'], percentiles['extracurricular'], percentiles['sleep_hours'],
            percentiles['assignments_completed']
        ]
    })

    fig = px.bar(pct_df, x='Feature', y='Percentile', title='Percentile rank vs historical students (higher is better)', range_y=[0,100])
    st.plotly_chart(fig, use_container_width=True)

    # Show yearly trend of average final grade and study_hours
    st.divider()
    st.header("📈 Historical Trends (Yearly averages)")

    trend_df = h.groupby('year').agg({
        'final_grade':'mean',
        'study_hours':'mean',
        'attendance':'mean'
    }).reset_index().round(2)

    # two charts side-by-side
    t1, t2 = st.columns(2)
    with t1:
        fig1 = px.line(trend_df, x='year', y='final_grade', markers=True, title='Average Final Grade by Year')
        st.plotly_chart(fig1, use_container_width=True)
    with t2:
        fig2 = px.line(trend_df, x='year', y='study_hours', markers=True, title='Average Study Hours by Year')
        st.plotly_chart(fig2, use_container_width=True)

    # Show nearest neighbors (simple distance in feature space) from historical data
    st.divider()
    st.header("🔎 Similar historical students (nearest by Euclidean on normalized features)")
    # normalize features by mean/std from historical subset
    feats = feature_names
    h_feats = h[feats].copy()
    mean = h_feats.mean()
    std = h_feats.std().replace(0,1)
    h_norm = (h_feats - mean) / std
    inp_norm = ((input_df[feats].iloc[0] - mean) / std).values
    distances = np.linalg.norm(h_norm.values - inp_norm, axis=1)
    idxs = np.argsort(distances)[:5]
    neighbors = h.iloc[idxs][feats + ['final_grade','performance','year']].copy()
    neighbors['distance'] = distances[idxs].round(3)
    st.dataframe(neighbors.reset_index(drop=True))

st.divider()
st.caption("🤖 This app uses a Random Forest trained on a synthetic 4-year dataset. Tweak weights & fusion logic as needed for your use case.")
