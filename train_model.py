import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("student_history.csv")

# FEATURES (roll_no NOT included)
features = [
    "study_hours",
    "attendance",
    "previous_grade",
    "sleep_hours",
    "assignments_completed",
   
]

X = df[features]
y = df["performance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "student_model.pkl")
print("Model trained & saved as student_model.pkl")
