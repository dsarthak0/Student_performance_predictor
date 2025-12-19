# 🎓 Student Performance Predictor

A Machine Learning–powered web application that predicts student academic performance using structured academic data.  
The system enables **early identification of at-risk students**, supports **data-driven academic decisions**, and improves overall educational outcomes.

---

## 📖 Table of Contents
- Project Overview  
- Motivation  
- Problem Statement  
- Key Features  
- Sustainable Development Goals (SDGs)  
- System Architecture  
- Machine Learning Pipeline  
- Dataset Description  
- Model Evaluation  
- Tech Stack  
- Installation & Execution  
- Results  
- Limitations  
- Future Scope  
- Contributors  

---

## 📌 Project Overview

The **Student Performance Predictor** is designed to analyze student academic behavior and predict future performance using **Machine Learning techniques**. Traditional academic evaluation methods are reactive and fail to identify struggling students early. This project introduces an **intelligent, automated, and explainable prediction system** to bridge that gap.

The application integrates a **Random Forest Classification model** with a **Flask-based web interface**, enabling real-time predictions based on student inputs.

---

## 💡 Motivation

Educational institutions generate vast amounts of academic data, yet much of it remains underutilized. Poor academic performance is often detected too late, when corrective measures are less effective.

This project aims to:
- Detect early warning signs of academic risk  
- Reduce failure and dropout rates  
- Assist teachers and counselors with predictive insights  
- Promote proactive rather than reactive academic intervention  

---

## ❓ Problem Statement

How can Machine Learning be effectively used to predict student academic performance using historical and real-time educational data in order to support early intervention and improve learning outcomes?

---

## ✨ Key Features

- 📊 Accurate student performance prediction  
- 🧠 Random Forest–based classification model  
- 🌐 Real-time web application using Flask  
- 🔍 Feature importance analysis for explainability  
- 📈 Handles historical and current academic data  
- 🧪 Synthetic data generation for class balancing  
- 🛡️ Input validation and error handling  

---

## 🌍 Sustainable Development Goals (SDGs)

This project contributes to the following **UN Sustainable Development Goals**:

- **SDG 4 – Quality Education** *(Primary)*  
  - Enhances learning outcomes through early performance prediction  
- **SDG 10 – Reduced Inequalities**  
  - Enables equitable academic support for all students  
- **SDG 3 – Good Health and Well-being** *(Indirect)*  
  - Reduces academic stress through early guidance  

---

## 🏗️ System Architecture

The system follows a **three-tier architecture**:

1. **Presentation Layer (Frontend)**  
   - User inputs student academic details  
2. **Application Layer (Backend – Flask)**  
   - Handles requests, validation, and preprocessing  
3. **Machine Learning Layer**  
   - Loads trained models and generates predictions  

---

## 🧠 Machine Learning Pipeline

1. Data Collection  
2. Data Cleaning & Preprocessing  
3. Feature Encoding & Normalization  
4. Train–Test Split  
5. Model Training (Random Forest)  
6. Model Evaluation  
7. Feature Importance Extraction  
8. Model Serialization (.pkl)  
9. Real-time Prediction via Flask  

---

## 📂 Dataset Description

### Datasets Used
- `student_data.csv` – Primary academic dataset  
- `student_history.csv` – Historical performance records  
- `data_generator.py` – Generates synthetic records  

### Key Attributes
- Study Hours  
- Attendance Percentage  
- Internal Exam Scores  
- Previous Academic Scores  
- Assignment Performance  

The target variable represents the **student performance category**.

---

## 📊 Model Evaluation

The model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

**Random Forest** was selected due to its:
- High accuracy  
- Robustness against overfitting  
- Ability to handle noisy data  
- Built-in feature importance support  

---

## 🛠️ Tech Stack

### Programming
- Python 3.x  

### Libraries
- NumPy  
- Pandas  
- Scikit-learn  

### Web Framework
- Flask  

### Tools
- VS Code / PyCharm  
- Jupyter Notebook  

---

## 🚀 Installation & Execution

### Clone Repository
```bash
git clone https://github.com/dsarthak0/Student_performance_predictor.git
cd Student_performance_predictor
````

### Install Dependencies

```bash
pip install numpy pandas scikit-learn flask
```

### Train the Model

```bash
python train_model.py
```

### Run Application

```bash
python app.py
```

### Open Browser

```
http://127.0.0.1:5000/
```

---

## 📈 Results

* High prediction accuracy
* Instant real-time predictions
* Stable performance across datasets
* Transparent decision-making through feature importance

---

## ⚠️ Limitations

* Works only with structured academic data
* Limited to single-institution usage
* No psychological or emotional analysis
* Static CSV-based data input

---

## 🔮 Future Scope

* LMS and ERP integration
* Cloud deployment (AWS / Render / Heroku)
* Deep Learning model integration
* Student recommendation engine
* Mobile application support
* Multi-institution scalability
* Automated model retraining

---

## 👥 Contributors

* **Sarthak Dhanwate**
* **Vansh Sutaria**
* **Adarsh Samani**
* **Siddhi Somaiya**

### 👩‍🏫 Project Guide

* **Ms. Aswathy S.**

---

## 🏫 Institution

**K. J. Somaiya College of Engineering**
Somaiya Vidyavihar University, Mumbai

