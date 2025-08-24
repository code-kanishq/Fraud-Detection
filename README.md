# Fraud Detection

This repository contains an end-to-end machine learning pipeline for detecting fraudulent financial transactions.  
The project was developed in **Google Colab** and leverages **XGBoost** for classification due to its high predictive performance on imbalanced datasets.

## 📌 Features
- **Data Loading & Preprocessing**  
  - Loaded dataset from Google Drive.  
  - Filtered relevant transaction types (`TRANSFER`, `CASH_OUT`).  
  - Created time-based and balance-related engineered features.  
  - One-hot encoding for categorical variables & standard scaling for numerical features.  

- **Exploratory Data Analysis (EDA)**  
  - Distribution of fraudulent vs. non-fraudulent transactions.  
  - Transaction type fraud analysis (confirmed fraud in `TRANSFER` and `CASH_OUT`).  
  - Visualizations using `matplotlib` and `seaborn`.  

- **Model Training**  
  - XGBoost Classifier with class imbalance handling (`scale_pos_weight`).  
  - Pipeline with preprocessing + model for reproducibility.  

- **Model Evaluation**  
  - Classification Report (Precision, Recall, F1-score).  
  - Confusion Matrix heatmap.  
  - ROC-AUC Score & ROC Curve.  
  - Precision-Recall Curve with Average Precision score.  

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn)  
- Scikit-learn (Pipeline, Preprocessing, Model Evaluation)  
- XGBoost  

## 📊 Results
- Achieved strong ROC-AUC and PR scores despite class imbalance.  
- XGBoost outperformed baseline approaches for fraud detection.  

---
