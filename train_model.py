# train_model.py
"""
Train a RandomForest model on the combined 4-year dataset and save:
- student_performance_model.pkl
- feature_names.json
- feature_importance.csv (for inspection)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os

np.random.seed(42)

DATA_PATH = 'data/students_2021_2024_combined.csv'
MODEL_PATH = 'student_performance_model.pkl'
FEATURES_JSON = 'feature_names.json'
IMPORTANCE_CSV = 'feature_importance.csv'

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Combined dataset not found at {path}. Run data_generator.py first.")
    df = pd.read_csv(path)
    return df

def train_and_save(df):
    feature_cols = [
        'study_hours', 'attendance', 'previous_grade',
        'parental_support', 'extracurricular', 'sleep_hours',
        'assignments_completed', 'age'
    ]
    X = df[feature_cols]
    y = df['performance']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and features
    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_JSON, 'w') as f:
        json.dump(feature_cols, f)

    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    fi.to_csv(IMPORTANCE_CSV, index=False)
    print(f"Saved model to {MODEL_PATH}, features to {FEATURES_JSON}, importance to {IMPORTANCE_CSV}")

    return model, fi

if __name__ == '__main__':
    df = load_data()
    model, fi = train_and_save(df)
