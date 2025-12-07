# train_model.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

DATA_PATH = "student_history.csv"
MODEL_PATH = "rf_grade_model.pkl"
FEATURES_JSON = "feature_names.json"

def construct_examples(df):
    rows = []
    # group by roll
    for roll, g in df.groupby("roll_no"):
        g_sorted = g.sort_values("year")
        years = g_sorted["year"].values
        # iterate year by year; for each year, use history from previous years only
        for idx in range(len(g_sorted)):
            if idx == 0:
                # no history -> skip (we need at least one past year to build historical aggregates)
                continue
            current_row = g_sorted.iloc[idx]
            past = g_sorted.iloc[:idx]  # all previous rows

            # historical aggregates (from all previous years)
            hist_mean_final = past["final_grade"].mean()
            hist_last_final = past["final_grade"].iloc[-1]
            # trend (slope) of final_grade vs year (if only one past year, slope = 0)
            if len(past) >= 2:
                slope = np.polyfit(past["year"].values, past["final_grade"].values, 1)[0]
            else:
                slope = 0.0
            hist_std_final = past["final_grade"].std(ddof=0) if len(past) >= 2 else 0.0
            hist_mean_study = past["study_hours"].mean()
            hist_mean_attendance = past["attendance"].mean()
            hist_count = len(past)

            # current year's observed inputs (these are the "current inputs" the user will supply at prediction time)
            study_hours = current_row["study_hours"]
            attendance = current_row["attendance"]
            previous_grade = current_row["previous_grade"]  # this is what was recorded at start of that year
            parental_support = current_row["parental_support"]
            extracurricular = current_row["extracurricular"]
            sleep_hours = current_row["sleep_hours"]
            assignments_completed = current_row["assignments_completed"]

            # target is the final_grade for this year
            target_final = current_row["final_grade"]

            feat_row = {
                "roll_no": roll,
                "year": int(current_row["year"]),
                # historical aggregates
                "hist_mean_final": hist_mean_final,
                "hist_last_final": hist_last_final,
                "hist_trend": slope,
                "hist_std_final": hist_std_final,
                "hist_mean_study": hist_mean_study,
                "hist_mean_attendance": hist_mean_attendance,
                "hist_count": hist_count,
                # current inputs
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_grade": previous_grade,
                "parental_support": parental_support,
                "extracurricular": extracurricular,
                "sleep_hours": sleep_hours,
                "assignments_completed": assignments_completed,
                # target
                "final_grade": target_final
            }
            rows.append(feat_row)
    examples = pd.DataFrame(rows)
    return examples

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run data_generator.py first.")
    df = pd.read_csv(DATA_PATH)
    examples = construct_examples(df)
    print(f"Constructed {len(examples)} training examples")

    feature_cols = [
        # historical
        "hist_mean_final", "hist_last_final", "hist_trend", "hist_std_final",
        "hist_mean_study", "hist_mean_attendance", "hist_count",
        # current inputs
        "study_hours", "attendance", "previous_grade",
        "parental_support", "extracurricular", "sleep_hours", "assignments_completed"
    ]

    X = examples[feature_cols].fillna(0.0)
    y = examples["final_grade"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    print("Training Random Forest regressor...")
    model.fit(X_train, y_train)

    # Eval
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # Save model + features
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_JSON, "w") as f:
        json.dump(feature_cols, f)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature list -> {FEATURES_JSON}")

if __name__ == "__main__":
    main()
