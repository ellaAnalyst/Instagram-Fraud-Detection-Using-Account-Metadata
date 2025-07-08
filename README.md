# 📊 Instagram User Type Detection: From Heuristics to ML Models

This project aims to classify Instagram accounts into four categories — **Real**, **Bot**, **Scam**, and **Spam** — based on their profile patterns, content activity, and behavioral ratios.

---

## 📌 Objectives

- Identify **suspicious accounts** (bots, scams, spam) from normal users
- Use both **heuristic rules** and **machine learning models**
- Understand key patterns through **EDA** and **feature engineering**
- Benchmark different classifiers to find the best model

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

- **Follow Ratio** = `Following / (Followers + 1)`
- **Post Ratio** = `Posts / (Followers + 1)`
- **Binary Flags**:
  - `NoBio`, `NoProfilePic`, `NoPosts`
- **Log-transformed ratios** to mitigate outlier effects
- **New hybrid features**: `NoBio_NoPic`, `NoPic_NoPost`

---

## 📈 EDA Insights

- `Bot` accounts have unusually high **Following** with near-zero posts
- `Scam` accounts have **high Follow Ratio** despite low Followers
- `Spam` accounts show **extremely high Follower counts**, distorting the mean
- **Binary Features** are useful for rule-based detection

📌 Visualization samples available in the `figures/` folder.

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
- High classification accuracy across all labels
- Slight confusion between `Real` and `Spam` classes

📁 See full classification report and confusion matrix in `figures/`

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

| File / Folder              | Description |
|---------------------------|-------------|
| `README.md`               | Project overview and key results |
| `Instagram_User_Type_Detection.ipynb` | Main Jupyter Notebook with full analysis |
| `figures/`                | Exported images (Confusion Matrix, Boxplots, Feature Importances) |
| `requirements.txt`        | Python libraries used |
| `.gitignore`              | Ignore temp files like `.ipynb_checkpoints/` |

---

## ⚙️ How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/yourusername/instagram-user-type-detection.git
   cd instagram-user-type-detection
