# 🧠 Student Score Prediction API

A Django REST Framework-based API that predicts **students’ final exam scores** using machine learning models (Neural Network, Linear Regression, and Random Forest).

This project preprocesses academic and behavioral data (attendance, study hours, assignments, internal marks, etc.) to forecast student performance — enabling early academic interventions.

---

## 📘 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset Description](#-dataset-description)
- [Model Training Process](#-model-training-process)
- [API Endpoint](#-api-endpoint)
- [Example Request and Response](#-example-request-and-response)
- [Model Evaluation Results](#-model-evaluation-results)
- [How to Run Locally](#-how-to-run-locally)
- [License](#-license)

---

## 📖 Overview

This system is part of a **Smart Student Management System**, focused on predicting student academic performance based on multiple factors such as:
- Study hours
- Attendance rate
- Internal and past exam marks
- Assignment submission rate
- Extracurricular activity participation

The prediction is powered by machine learning models trained in Python using **TensorFlow**, **Scikit-learn**, and integrated with **Django REST Framework** for API exposure.

---

## 🚀 Features

✅ Predict student final exam scores from JSON input  
✅ Uses multiple models (Neural Network, Random Forest, Linear Regression)  
✅ Well-preprocessed input data with scaling and encoding  
✅ REST API for real-time predictions  
✅ Easy integration with any front-end or mobile app  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend Framework** | Django|
| **API Framework** | Django REST Framework (DRF) |
| **Machine Learning** | TensorFlow / Keras, Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Database** | SQLite |
| **Visualization** | Matplotlib, Seaborn |

---

## 📂 Project Structure

student_score_prediction/
│
├── score_predictor/
| ├──migrations/
│ ├── models.py
│ ├── serializers.py # Input and output serializers
│ ├── views.py # Main API logic for prediction
│ ├── urls.py # API route configuration
│
├──config/
| ├── settings.py
| ├── urls.py
| ├── asgi.py
| ├── wsgi.py
|
├── student_performance_dataset.csv
├── student_score_model.h5 # Trained Neural Network model
├── manage.py
└── README.md


---

## 🧮 Dataset Description

| Column | Description | Example |
|--------|--------------|----------|
| `Gender` | 0 = Male, 1 = Female | 0 |
| `Study_Hours_per_Week` | Hours of study per week | 15 |
| `Attendance_Rate` | Attendance percentage | 0.85 |
| `Past_Exam_Scores` | Average of past exam scores | 78 |
| `Parental_Education_Level` | 0 = High School, 1 = Bachelors, 2 = Masters, 3 = PhD | 2 |
| `Internet_Access_at_Home` | 0 = No, 1 = Yes | 1 |
| `Extracurricular_Activities` | 0 = No, 1 = Yes | 1 |
| `Internal_Marks` | Internal performance marks (scaled) | 0.88 |
| `Assignment_Submission_Rate` | Submission percentage | 0.90 |
| `Internal_Assessment_Marks` | Weighted internal assessment score | 0.84 |
| `Final_Exam_Score` | Target variable (actual marks) | 82 |

---

## 🧠 Model Training Process

The dataset undergoes the following preprocessing:

1. **Feature scaling** using `MinMaxScaler()`  
2. **Categorical encoding** for gender, education, etc.  
3. **Feature engineering**:
   - Internal Marks = 0.6 × Past Exam + 0.4 × Study Hours
   - Internal Assessment = 0.5 × Internal Marks + 0.5 × Past Exam
4. **Model Training**:
   - **Neural Network (TensorFlow)**
   - **Linear Regression (Scikit-learn)**
   - **Random Forest Regression (Scikit-learn)**

The trained neural network (`student_score_model.h5`) is then loaded into Django for prediction.

---

## 🌐 API Endpoint

### **POST /api/score_prediction/**

Predicts the final exam score for one or more students based on provided academic and behavioral data.

---

## 📦 Example Request and Response

### 🧾 Request

**POST** `http://localhost:8000/api/score_prediction/`

```json
{
    "Gender": 0,
    "Study_Hours_per_Week": 0.8,
    "Attendance_Rate": 0.9,
    "Past_Exam_Scores": 0.85,
    "Parental_Education_Level": 2,
    "Internet_Access_at_Home": 1,
    "Extracurricular_Activities": 1,
    "Internal_Marks": 0.88,
    "Assignment_Submission_Rate": 0.92,
    "Internal_Assessment_Marks": 0.86
}
Response
{
    "message": "Score prediction successful",
    "predicted_final_scores": [0.84]
}

```

---

## ⚙️ Installation (Run Locally)

Follow these steps to set up the project on your local machine 👇

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/student-score-prediction.git
cd student-score-prediction
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
pip install django djangorestframework tensorflow scikit-learn pandas numpy matplotlib seaborn
pip freeze > requirements.txt
python manage.py runserver

curl -X POST http://127.0.0.1:8000/api/score_prediction/ \
-H "Content-Type: application/json" \
-d '{
  "Gender": 1,
  "Study_Hours_per_Week": 0.82,
  "Attendance_Rate": 0.95,
  "Past_Exam_Scores": 0.80,
  "Parental_Education_Level": 3,
  "Internet_Access_at_Home": 1,
  "Extracurricular_Activities": 1,
  "Internal_Marks": 0.85,
  "Assignment_Submission_Rate": 0.90,
  "Internal_Assessment_Marks": 0.88
}'


```