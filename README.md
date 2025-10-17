# TruScan
Based on the files you uploaded—specifically the Jupyter Notebook (`Rumour Detector.ipynb`), the Python script (`Rumour_Detector.py`), and the CSV file (`fake.csv`)—your project uses a **Logistic Regression** model with **TF-IDF** vectorization for rumor detection.

Here is a comprehensive `README.md` file tailored to your repository.

-----

# 🕵️ Rumour Detector: NLP-Based Fake News Classification

> An NLP-powered machine learning pipeline that classifies text as either **Credible (Real News)** or **Rumour (Fake/Biased News)** using **TF-IDF** and **Logistic Regression**.

## 🌟 Overview

This project implements a text classification system designed to quickly identify and flag potentially misleading or biased content. The model is trained on a dataset containing news articles labeled by type (including 'bs', 'conspiracy', 'fake', etc.) to learn patterns that distinguish reliable content from rumors.

-----

## 🚀 Getting Started

### Prerequisites

You need **Python 3.x** and the required libraries installed via `pip`.

-----

## 💡 Usage

The project can be run directly using the Jupyter Notebook for step-by-step analysis, or via the Python script if you deploy it as a Streamlit application (based on the imports in `Rumour_Detector.py`).

### 1\. Run the Analysis Notebook

To see the full data cleaning, preprocessing, training, and evaluation pipeline:

1.  Start your Jupyter server:
    ```bash
    jupyter notebook
    ```
2.  Open the file: **`Rumour Detector.ipynb`**
3.  Run all cells sequentially.

### 2\. Model Training and Prediction

The core pipeline, found in both the notebook and the `.py` file, follows these steps:

1.  **Data Loading:** Loads the **`fake.csv`** dataset.
2.  **Preprocessing:** Combines the `author` and `title` columns into a single `content` column.
3.  **Labeling:** Creates the binary `label` column where:
      * **0 (Credible)**: Content where the `type` column is NOT in the keyword list (`bias`, `conspiracy`, `fake`, `bs`, `hate`).
      * **1 (Rumour)**: Content where the `type` column **IS** in the keyword list.
4.  **Stemming & Cleaning:** Applies the `PorterStemmer` and removes stopwords.
5.  **Vectorization:** Uses **`TfidfVectorizer`** to convert text into numerical feature vectors.
6.  **Classification:** Trains a **`LogisticRegression`** model.

-----

## 📊 Technical Details

| Component | Detail |
| :--- | :--- |
| **Model** | **Logistic Regression** |
| **Feature Extraction** | **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) |
| **Primary Data File**| `fake.csv` |
| **Programming Language** | Python 3.x |
| **Key Libraries** | `pandas`, `scikit-learn`, `nltk`, `re` |

### Expected Performance

*Based on the snippet from your notebook output, the model achieves high accuracy:*

  * **Training Accuracy:** ≈ **97.27%**

-----

## 📦 Repository Structure

```
[project-root]/
├── Rumour Detector.ipynb  # Main notebook with the complete ML workflow.
├── Rumour_Detector.py     # Python script (likely for deployment/Streamlit).
├── fake.csv               # The raw dataset used for training.
└── README.md              # This file.



```

