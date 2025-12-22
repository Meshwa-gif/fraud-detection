# Fraud Detection Using Machine Learning

This repository contains an end-to-end machine learning workflow for detecting fraudulent credit card transactions.  
The notebook demonstrates data loading, validation, exploratory analysis, model training, and evaluation for a **highly imbalanced** classification problem.

➡️ Main notebook: `notebooks/fraud_detection.ipynb`

---

## Dataset

- Kaggle dataset: **mlg-ulb/creditcardfraud**
- Target column: `Class` (1 = Fraud, 0 = Non-Fraud)
- Features: `Time`, `Amount`, and PCA-transformed variables `V1`–`V28`

In the notebook, the dataset is downloaded programmatically using **kagglehub**.

---

## What’s Inside the Notebook

### 1) Data Loading & Validation
- Loads `creditcard.csv` via `kagglehub.dataset_download(...)`
- Confirms dataset shape and checks for missing values

### 2) EDA
- Visualizes class imbalance
- Examines transaction amount distribution
- Uses imbalance-aware evaluation thinking (accuracy alone is misleading)

### 3) Modeling (3 models)
- **Logistic Regression (baseline)**
- **Logistic Regression with `class_weight="balanced"`**
- **Random Forest with `class_weight="balanced_subsample"`**

### 4) Evaluation
- Classification report (precision/recall/F1)
- Confusion matrix
- ROC curve + ROC AUC

---

## Results (From the Notebook)

Fraud class (Class = 1) performance on the test set:

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC AUC |
|------|-------------------:|---------------:|-----------:|--------:|
| Logistic Regression | 0.85 | 0.68 | 0.76 | 0.9484 |
| Balanced Logistic Regression | 0.06 | 0.92 | 0.11 | 0.9722 |
| Random Forest | **0.96** | **0.74** | **0.84** | 0.9471 |

**Interpretation:**  
- Class-weighting greatly increases recall but causes many false positives (low precision).  
- Random Forest provides the best balance between precision and recall for practical fraud detection.

---

## How to Run

### Option A (Matches the notebook): KaggleHub download
1. Install dependencies
2. Ensure Kaggle access is configured for KaggleHub (Kaggle credentials/environment)
3. Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook
