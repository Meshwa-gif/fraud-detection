# Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent financial transactions using supervised machine learning techniques.  
It demonstrates an **end-to-end data science workflow**, including data validation, exploratory analysis, model training, evaluation, and interpretability, with emphasis on handling **severely imbalanced data**.

---

## Problem Statement

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of total activity.  
The objective of this project is to build and evaluate models that can **accurately identify fraudulent transactions while balancing false positives and false negatives**, a critical trade-off in real-world fraud systems.

---

##  Dataset

- **Source:** Credit Card Fraud Dataset (Kaggle)  
- **Observations:** 284,807 transactions  
- **Features:** 30 (PCA-transformed variables `V1–V28`, `Time`, `Amount`)  
- **Target Variable:** `Class` (1 = Fraud, 0 = Non-Fraud)

> Due to size and licensing restrictions, the dataset is **not included** in this repository.

---

## Data Preparation & Validation

- Verified dataset integrity (no missing values)
- Examined feature distributions and class imbalance
- Separated features and target to avoid data leakage
- Used **stratified train-test split** to preserve fraud distribution

---

## Exploratory Data Analysis (EDA)

Key insights from EDA:
- Fraudulent transactions account for **less than 1%** of all observations
- Transaction amounts are **highly right-skewed**
- Accuracy alone is misleading; **recall, precision, F1-score, and ROC AUC** are more appropriate metrics

---

## Modeling Approach

Three models were trained and evaluated:

### Logistic Regression (Baseline)
- Interpretable linear model
- High precision but moderate recall
- Serves as a baseline reference

### Logistic Regression (Class-Weighted)
- Applied `class_weight="balanced"`
- Significantly improved recall
- Introduced a high number of false positives

### Random Forest (Final Model)
- Captures non-linear patterns
- Achieved the **best balance between precision and recall**
- Strong overall performance across metrics

---

##  Model Performance Summary

| Model | Precision (Fraud) | Recall (Fraud) | F1-score | ROC AUC |
|-----|------------------|---------------|---------|---------|
| Logistic Regression | High | Moderate | Moderate | ~0.96 |
| Balanced Logistic Regression | Very Low | Very High | Low | ~0.97 |
| **Random Forest (Selected)** | **0.96** | **0.74** | **0.84** | **~0.95** |

---

##  Final Model Selection

**Random Forest** was selected as the final model because it:
- Captures most fraudulent cases
- Maintains a low false positive rate
- Balances fraud prevention with operational cost
- Demonstrates strong real-world applicability

---

## Feature Importance & Interpretability

Feature importance analysis was performed using:
- Built-in Random Forest feature importance
- Permutation importance (ROC AUC–based)

Top contributing features (e.g., `V14`, `V10`, `V17`) consistently ranked highly across methods.

> Since features are PCA-transformed, importance reflects abstract transaction patterns rather than directly interpretable raw attributes.

---

## Model Limitations & Bias

- Severe class imbalance can bias predictions toward non-fraud cases
- False positives may incur operational review costs
- The model relies on historical patterns and may not detect novel fraud strategies
- Predictions should be used as **decision support**, not fully automated decisions

These limitations highlight the importance of **human-in-the-loop validation** in AI-driven fraud detection systems.

---

## How to Run the Project

1. Clone this repository
2. Download the Credit Card Fraud Dataset from Kaggle
3. Place `creditcard.csv` inside the `data/` folder
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
