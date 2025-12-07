import pandas as pd
import numpy as np

np.random.seed(42)

# 50 roll numbers
roll_numbers = np.arange(101, 151)

# Generate 3 years of history for each student
years = [2021, 2022, 2023]

rows = []
for roll in roll_numbers:
    for yr in years:
        study = np.round(np.random.uniform(2, 10), 1)
        attendance = np.random.randint(50, 100)
        prev_grade = np.random.randint(40, 95)
        sleep = np.round(np.random.uniform(5, 9), 1)
        assignments = np.random.randint(5, 10)
        # age = np.random.randint(15, 18)

        # Final grade slightly influenced by input
        final_grade = int(
            study / 10 * 20 +
            attendance / 100 * 25 +
            prev_grade / 100 * 35 +
            sleep / 10 * 10 +
            assignments / 10 * 10 +
            np.random.randint(-5, 5)
        )

        performance = (
            "High" if final_grade >= 80 else
            "Medium" if final_grade >= 60 else
            "Low"
        )

        rows.append([roll, yr, study, attendance, prev_grade,
                     sleep, assignments, final_grade, performance])

df = pd.DataFrame(rows, columns=[
    "roll_no", "year", "study_hours", "attendance",
    "previous_grade", "sleep_hours", "assignments_completed",
     "final_grade", "performance"
])

df.to_csv("student_history.csv", index=False)
print("Generated student_history.csv with 50 roll numbers × 3 years")
