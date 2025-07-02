# 🧠 Credit Card Default Prediction from Natural Language Queries

Predict credit card default risk directly from plain English using LLMs + deep tabular learning.

> Example input:  
> “A 28-year-old man with a limit of 20,000 and late on his last 4 payments. Bills are rising and he pays only the minimum.”

🔗 <video src="credit_default_demo.mp4" controls="controls" width="100%" />

## 🚀 Overview

This project turns a sentence into a risk score using:
- 🧠 **Claude 3 (Anthropic)**: Parses free-text into structured features
- 🔄 **Feature Engineering**: Binning, WOE, ratios, and aggregations
- ⚖️ **SMOTE**: For class imbalance
- 🎯 **Model Training**: Logistic Regression, Random Forest, XGBoost, Meta Learners, FT-Transformer
- 📈 **Hyperparameter Tuning**: GridSearchCV + Optuna
- 🧪 **Cross-Validation**: Robust metrics (AUC, precision, recall, F1)
- 💻 **Gradio UI**: Predict defaults from natural language, instantly

---

## 📊 Dataset

Based on the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients):

- ~23 features including age, payment history, bill amounts, and credit limits
- Highly imbalanced target (default vs non-default)

---

## 🔧 Feature Engineering

✅ Log-transformations (e.g., `LIMIT_BAL_log`)  
✅ Binning + Weight of Evidence (WOE) encoding  
✅ Aggregations: average bill/payment, ratios (e.g., `PAY_AMT_RATIO`)  
✅ Delta calculations (e.g., `PAY_DIFF` = avg bill - avg payment)

---

## 🧪 Models Implemented

| Model                         | Preprocessing        | Tuning       | CV Supported | Notes                        |
|------------------------------|----------------------|--------------|--------------|------------------------------|
| Logistic Regression          | One-Hot + Scaling    | GridSearchCV | ✅           | Baseline + WOE variant       |
| Random Forest                | One-Hot              | GridSearchCV | ✅           | Higher recall                |
| XGBoost                      | One-Hot              | GridSearchCV | ✅           | Good AUC, fast inference     |
| Meta Learner (Stacked)       | Ensemble             | Manual       | ✅           | Base: LR + RF + XGB          |
| Calibrated Meta Learner      | Ensemble + Prob Cal. | Manual       | ✅           | Improves confidence outputs  |
| **FT-Transformer**           | One-Hot + Scaling    | Optuna       | ✅           | Deep tabular model           |
| **FT-Transformer (Deep)**    | + Dropout, 100 Epochs| Optuna       | ✅           | Best performer               |

---

## 📉 Performance Summary

### 🔹 FT-Transformer (SMOTE + Deep + Optuna)
- AUC: **0.8631**
- Accuracy: **77.9%**
- Precision: **77.6%**
- Recall: **78.4%**
- F1 Score: **0.780**
- Confusion Matrix:  
  - TP = 2422  
  - TN = 2390  
  - FP = 699  
  - FN = 666

### 🔹 TabTransformer (Baseline)
- AUC: **0.7608**
- Accuracy: **80.5%**
- Precision: **62.1%**
- Recall: **37.8%**
- F1 Score: **0.470**

---

## ⚠️ Error Sensitivity: FP vs FN

- **False Positive (FP)**: Model flags someone as risky who isn’t → loss of good customer.
- **False Negative (FN)**: Model misses a defaulter → financial loss to lender.

> A strong model balances high precision (fewer FPs) and high recall (fewer FNs).

---

## 💬 Gradio Interface (LLM-Powered)

- 🧠 Claude extracts structured features from plain English
- 🎯 Passes to trained model (FT-Transformer)
- 📊 Returns predicted risk and confidence

```python
# Example query
"A 45-year-old woman, credit limit 30,000, delayed past 3 payments, pays ~500 monthly."

→ Output: Default Probability: 72.4%
```


📂 Project Structure

📁 CreditCardDefaulters
```
├── data/
│   ├── Credit_Card_train.csv
├── models/
│   ├── ft_transformer_model.h5
├── notebooks/
│   ├── eda_feature_engineering.ipynb
│   ├── modeling_gridsearch_optuna.ipynb
├── app.py           # Gradio Interface
├── utils.py         # Claude API call + feature extractor
└── README.md
```



