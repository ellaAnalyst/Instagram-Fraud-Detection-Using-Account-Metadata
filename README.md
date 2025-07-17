# 📊 Instagram User Type Detection: From Heuristics to ML Models

This project aims to classify Instagram accounts into four categories — **Real**, **Bot**, **Scam**, and **Spam** — based on their profile patterns, content activity, and behavioral ratios.

---

## 📌 Objectives

- Detect **suspicious accounts** using both rules-based heuristics and supervised learning
- Perform structured **feature engineering** for pattern discovery
- Use **EDA** to understand behavioral signals across account types
- Benchmark classifiers to identify the most robust model

---

## 🧩 Dataset Overview

- **Total Accounts**: 15,000  
- **Labels**: `Real`, `Bot`, `Scam`, `Spam`
- **Features**:
  - Follower/Following/Post counts
  - Bio presence, Profile Picture presence, Post existence
  - Derived ratios (Follow Ratio, Post Ratio)
  - Combined binary features (e.g. `NoBio & NoPic`)

---

## 🛠️ Feature Engineering Highlights

- **Follow Ratio** = Following / (Followers + 1)
- **Post Ratio** = Posts / (Followers + 1)
- **Binary Flags**: NoBio, NoProfilePic, NoPosts
- Log-transformed ratios to handle skewness
- Hybrid binary features for behavioral pattern amplification


---

## 📈 Key EDA Insights

- `Bot` accounts: High **Following**, near-zero posting
- `Scam` accounts: Inflated **Follow Ratios**, minimal followers
- `Spam` accounts: Abnormally high **Follower counts**
- Binary heuristics effective for initial detection layers

> Visualization samples provided in the `figures/` folder.
---

## 🤖 Model Benchmarking

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 94.0%    | 94.1%     | 94.0%  | 94.0%    |
| Decision Tree      | 96.2%    | 96.2%     | 96.2%  | 96.2%    |
| 🌟 Random Forest    | **96.3%**| **96.3%** | **96.3%**| **96.3%**|

✔️ **Final model**: `RandomForestClassifier`

---

## 📊 Final Model Performance

- Overall Accuracy: **96%**
- Stable classification performance across all categories  
- Slight confusion between `Real` and `Spam` accounts (behavioral overlap)  
- Model interpretability via feature importances

---

## 🔍 Key Feature Importance

Top predictors identified:

1. **Followers**
2. **Posts**
3. **Following**
4. **Follow Ratio**
5. **NoBio**, **NoPosts**

---

## 📂 Repository Structure

| File / Folder         | Description                                    |
|-----------------------|------------------------------------------------|
| `README.md`           | Project overview and key results               |
| `notebook/`           | Jupyter Notebook with full analysis            |
| `models/`             | Saved trained Random Forest model (.pkl file)  |
| `figures/`            | Exported images (confusion matrix, plots, etc.)|
| `requirements.txt`    | Python libraries used                          |


## 📦 Model Delivery

Final trained model saved in:  
`models/final_random_forest_model.pkl`

Ready for inference or integration into detection pipelines.

---

<p align="center">
  🎯 A data-driven approach to explainable Instagram account classification.
</p>
