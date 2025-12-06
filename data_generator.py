# data_generator.py
"""
Generate realistic synthetic student datasets for 4 years (2021-2024).
Each CSV will contain the exact features you required:
- study_hours
- attendance
- previous_grade
- parental_support
- extracurricular
- sleep_hours
- assignments_completed
- age
- year
- final_grade
- performance (Low/Medium/High)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_year_data(year, n_samples=2000):
    data = {
        'study_hours': np.random.normal(6, 2.5, n_samples).clip(0, 12).round(1),
        'attendance': np.random.normal(80, 14, n_samples).clip(0, 100).round(1),
        'previous_grade': np.random.normal(68, 16, n_samples).clip(0, 100).round(1),
        'parental_support': np.random.choice([1,2,3,4,5], n_samples, p=[0.06, 0.14, 0.35, 0.30, 0.15]),
        'extracurricular': np.random.choice([0,1,2,3,4,5], n_samples, p=[0.10,0.20,0.30,0.25,0.10,0.05]),
        'sleep_hours': np.random.normal(7, 1.2, n_samples).clip(4, 12).round(1),
        'assignments_completed': np.random.normal(78, 18, n_samples).clip(0,100).round(1),
        'age': np.random.choice([15,16,17,18,19], n_samples, p=[0.15,0.25,0.30,0.20,0.10]),
    }
    df = pd.DataFrame(data)
    df['year'] = year

    # Weighted performance score (same formula style as you used)
    perf_score = (
        df['study_hours'] / 12 * 25 +
        df['attendance'] / 100 * 20 +
        df['previous_grade'] / 100 * 20 +
        df['parental_support'] / 5 * 15 +
        df['extracurricular'] / 5 * 10 +
        df['sleep_hours'] / 12 * 5 +
        df['assignments_completed'] / 100 * 5
    )

    # add mild noise and clip
    perf_score += np.random.normal(0, 3.0, len(df))
    perf_score = perf_score.clip(0, 100)

    # categorize into Low/Medium/High using quantiles per-year for balance
    df['performance'] = pd.qcut(perf_score, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

    # final_grade correlated with perf_score and previous_grade
    df['final_grade'] = (perf_score * 0.7 + df['previous_grade'] * 0.3 + np.random.normal(0, 5, len(df))).clip(0, 100).round(1)

    return df

def main(output_dir='data', years=[2021,2022,2023,2024], n_per_year=2000):
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []
    for year in years:
        df_year = generate_year_data(year, n_samples=n_per_year)
        filename = os.path.join(output_dir, f'students_{year}.csv')
        df_year.to_csv(filename, index=False)
        print(f"Saved {len(df_year)} rows to {filename}")
        all_dfs.append(df_year)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_file = os.path.join(output_dir, 'students_2021_2024_combined.csv')
    combined.to_csv(combined_file, index=False)
    print(f"Saved combined dataset {combined_file} ({len(combined)} rows)")

if __name__ == '__main__':
    main()
