# 🍽️ Restaurant Review Sentiment Analysis (NLP & Machine Learning)

An end-to-end Natural Language Processing (NLP) pipeline designed to classify customer restaurant reviews into positive or negative sentiments. 

This project demonstrates the full lifecycle of a text classification task—from raw text preprocessing and vectorization to diagnosing and fixing **High Variance (Overfitting)** using model tuning techniques.

---

## 📌 Project Overview
* **Domain:** Natural Language Processing / Machine Learning
* **Goal:** Automatically analyze textual feedback to determine sentiment.
* **Key Challenge Solved:** Addressed high overfitting where an initial unpruned Decision Tree memorized training data (99.6% train accuracy vs. 71% test accuracy).
* **Final Solution:** Optimized the pipeline using a hyperparameter-tuned **Random Forest Classifier**, achieving a generalized **Best Fit** model (86.0% train accuracy vs. 75.0% test accuracy).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Processing:** `Pandas`, `NumPy`, `re` (Regex)
* **NLP Techniques:** `nltk` (Stopwords filtering, PorterStemmer), `TF-IDF Vectorizer`
* **Machine Learning:** `Scikit-Learn` (DecisionTreeClassifier, RandomForestClassifier)
* **Evaluation Metrics:** Accuracy Score, Confusion Matrix

---

## ⚙️ Model Performance & Optimization

| Model Architecture | Train Accuracy (Bias) | Test Accuracy (Variance) | Status |
| :--- | :---: | :---: | :---: |
| **Decision Tree (Unpruned)** | 99.6% | 71.0% | ❌ Overfitting / High Variance |
| **Random Forest (`max_depth=10`)** | **86.0%** | **75.0%** | ✅ Best Fit / Generalized |

---

## 📁 Repository Structure
```text
├── Restaurant_Reviews.tsv      # Dataset (1,000 labeled reviews)
├── main.py                     # Main Python script containing NLP pipeline
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
