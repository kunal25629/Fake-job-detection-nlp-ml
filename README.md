# Fake-job-detection-nlp-ml
Fake Job Detection using NLP, TF-IDF, and Machine Learning models (XGBoost, Random Forest, Logistic Regression) with Streamlit deployment.

# 🕵️ Fake Job Detection Using Machine Learning & NLP

## 📌 Project Overview

Fake job postings have become a significant issue on online job portals, leading to scams, identity theft, and financial fraud. This project leverages **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically identify whether a job posting is genuine or fraudulent.

The model analyzes job-related information such as job descriptions, company profiles, requirements, benefits, and other attributes to classify job listings as **Real** or **Fake**.

---

## 🎯 Problem Statement

Online recruitment platforms contain thousands of job advertisements. While many are legitimate, some are fake and designed to deceive job seekers.

The objective of this project is to build an intelligent machine learning system capable of detecting fraudulent job postings and helping users make safer career decisions.

---

## 📂 Dataset

Dataset: Fake Job Postings Dataset

### Features Include:
- Job Title
- Company Profile
- Description
- Requirements
- Benefits
- Employment Type
- Industry
- Function
- Location
- Salary Range
- Fraudulent Label (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Matplotlib
- Seaborn
- Streamlit

---

## 🔍 Project Workflow

### 1. Data Collection
- Loaded and explored the dataset.

### 2. Data Preprocessing
- Handled missing values.
- Removed irrelevant features.
- Cleaned textual data.

### 3. NLP Pipeline
- Text Lowercasing
- Removal of Special Characters
- Removal of Extra Spaces
- Text Normalization
- TF-IDF Vectorization

### 4. Exploratory Data Analysis (EDA)
- Fraudulent vs Genuine Job Distribution
- Correlation Analysis
- Feature Insights

### 5. Model Building
Implemented and compared:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### 6. Model Evaluation
Evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- ROC-AUC Curve

### 7. Cross Validation
Performed K-Fold Cross Validation to verify model robustness.

### 8. Hyperparameter Tuning
Optimized model performance using GridSearchCV.

### 9. Model Deployment
Built an interactive web application using Streamlit.

---

## 📊 Results

After comparing multiple machine learning models:

| Model | Performance |
|---------|-------------|
| Logistic Regression | Good |
| Random Forest | Better |
| XGBoost | Best |

✅ XGBoost achieved the highest overall performance and was selected as the final model for deployment.

---

## 🚀 Features

- Detects fake job postings automatically.
- NLP-based text processing pipeline.
- TF-IDF feature extraction.
- Multiple model comparison.
- Cross-validation and hyperparameter tuning.
- Streamlit web application for real-time predictions.

---

## 📁 Project Structure

```text
Fake-Job-Detection/
│
├── app.py
├── fake_job_detector.pkl
├── fake_job_postings.csv
├── Project Phase 1.ipynb
├── Project Phase 2.ipynb
├── requirements.txt
├── README.md
│
└── assets/
