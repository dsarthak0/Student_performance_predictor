# data_generator.py
import numpy as np
import pandas as pd
import os

np.random.seed(42)

ROLLS = np.arange(101, 151)  # 50 students
YEARS = [2021, 2022, 2023, 2024]

def generate_row(roll, year):
    # base personal tendency per student for some realism
    base_skill = np.random.normal(70, 8)  # baseline quality
    study_hours = np.round(np.random.normal(6, 2), 1)
    study_hours = float(np.clip(study_hours, 0, 12))
    attendance = float(np.clip(np.random.normal(80, 10), 40, 100))
    previous_grade = float(np.clip(base_skill + np.random.normal(0, 6), 30, 100))
    parental_support = int(np.random.choice([1,2,3,4,5], p=[0.06,0.14,0.35,0.30,0.15]))
    extracurricular = int(np.random.choice([0,1,2,3,4,5], p=[0.10,0.20,0.30,0.25,0.10,0.05]))
    sleep_hours = float(np.round(np.random.normal(7, 1.2), 1))
    assignments_completed = float(np.clip(np.random.normal(78, 15), 0, 100))

    # weighted score used to compute final grade (same as your earlier idea)
    perf_score = (
        study_hours / 12 * 25 +
        attendance / 100 * 20 +
        previous_grade / 100 * 20 +
        parental_support / 5 * 15 +
        extracurricular / 5 * 10 +
        sleep_hours / 12 * 5 +
        assignments_completed / 100 * 5
    )

    perf_score += np.random.normal(0, 4)  # noise
    final_grade = float(np.clip(perf_score, 0, 100).round(1))

    performance = "High" if final_grade >= 80 else ("Medium" if final_grade >= 60 else "Low")

    return {
        "roll_no": int(roll),
        "year": int(year),
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_grade": previous_grade,
        "parental_support": parental_support,
        "extracurricular": extracurricular,
        "sleep_hours": sleep_hours,
        "assignments_completed": assignments_completed,
        "final_grade": final_grade,
        "performance": performance
    }

def main(output_dir=".", n_per_year_per_roll=1):
    rows = []
    for roll in ROLLS:
        # add per-roll baseline variation by setting a seed dependency on roll
        for year in YEARS:
            row = generate_row(roll, year)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "student_history.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
