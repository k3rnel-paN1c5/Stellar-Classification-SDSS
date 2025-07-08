# Stellar Classification: A Machine Learning Approach

This project uses machine learning to classify celestial objects from the Sloan Digital Sky Survey (SDSS) as either a **Galaxy**, **Star**, or **Quasar (QSO)**. Four different classification models are built, tuned, and evaluated to determine the most effective approach for this astronomical dataset.


---

## 📋 Table of Contents
1. [Project Goal](#-project-goal)
2. [Dataset](#-dataset)
3. [Methodology](#-methodology)
4. [Model Performance](#-model-performance)
5. [Conclusion & Key Findings](#-conclusion--key-findings)
6. [How to Run](#-how-to-run)
7. [Dependencies](#-dependencies)

---

## 🎯 Project Goal
The primary objective is to build and evaluate several machine learning models to accurately classify celestial objects based on their spectral and positional data. The project emphasizes a full machine learning workflow, from data cleaning and feature engineering to hyperparameter tuning and model evaluation.

---

## 📊 Dataset
The data is sourced from the **Sloan Digital Sky Survey (SDSS) Data Release 18**. Each observation consists of 17 feature columns describing the object's position (RA, Dec), spectral measurements (photometric filter bands `u, g, r, i, z`), and redshift.

The target variable is the `class` column, which categorizes each object as a "GALAXY", "STAR", or "QSO". The dataset contains 100,000 observations.

---

## 🛠️ Methodology

The project follows a standard machine learning workflow:

1.  **Data Preparation & Cleaning**:
    * Loaded the dataset and handled the categorical `class` column by mapping it to numerical values (0: GALAXY, 1: STAR, 2: QSO).
    * Removed irrelevant identifier columns (`obj_ID`, `run_ID`, etc.) that do not contribute to the physical classification of the objects.

2.  **Feature Engineering & EDA**:
    * An initial correlation analysis revealed high multicollinearity between the photometric filter bands (`u, g, r, i, z`).
    * Engineered new, more informative features called **color indices** (e.g., `u-g`, `g-r`) by calculating the difference between adjacent filter bands. This is a common practice in astronomy that often yields more predictive power than raw magnitudes.
    * Dropped the original, redundant filter bands and the `MJD` (observation date) column.

3.  **Model Training & Hyperparameter Tuning**:
    * The data was split into a 70% training set and a 30% testing set, using **stratification** to maintain the original class distribution in both sets due to class imbalance.
    * Built `scikit-learn` **Pipelines** for each model to chain data scaling and classification, preventing data leakage.
    * Used `GridSearchCV` with 5-fold cross-validation to systematically tune the hyperparameters for four different classification models:
        * **Decision Tree**
        * **K-Nearest Neighbors (KNN)**
        * **Gaussian Naive Bayes**
        * **Support Vector Machine (SVM)**

4.  **Model Evaluation**:
    * Rigorously evaluated each tuned model's performance on both the training and testing data using accuracy, precision, recall, and F1-score.
    * A final comparison was made to select the best overall model.

---

## 🚀 Model Performance

The final performance of the tuned models on the unseen test set is summarized below:

| Model | Accuracy | Weighted Avg F1-Score | Weighted Avg Precision | Weighted Avg Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | **0.9753** | **0.9752** | **0.9751** | **0.9753** |
| SVM | 0.9722 | 0.9720 | 0.9721 | 0.9722 |
| K-Nearest Neighbors | 0.9548 | 0.9547 | 0.9548 | 0.9548 |
| Naive Bayes | 0.7418 | 0.6743 | 0.7923 | 0.7418 |

---

## ✨ Conclusion & Key Findings

The **Decision Tree classifier is the best-performing model** for this task, with a test accuracy and F1-score of **97.5%**.

* **Why the Decision Tree Excelled**: The model's hierarchical, rule-based structure was perfectly suited to the data. It effectively learned that **redshift** is the most powerful feature for separating stars (near-zero redshift) from galaxies and QSOs (high redshift). It then used the engineered **color indices** to accurately distinguish between the remaining classes.

* **Primary Source of Error**: The main challenge for all models was distinguishing between **galaxies and QSOs**. This is an astronomically reasonable difficulty, as a quasar is an active nucleus *within* a host galaxy, and some compact, blue star-forming galaxies can mimic the spectral characteristics of a QSO.

* **Why Naive Bayes Underperformed**: The Naive Bayes model performed poorly (74.2% accuracy) because its core assumption of feature independence is violated in this dataset (the color indices are inherently correlated). This led to a significant failure in identifying stars, as the model could not prioritize the overwhelming evidence from the redshift feature.

---

## ▶️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/k3rnel-paN1c5/stellar-classification.git](https://github.com/k3rnel-paN1c5/stellar-classification.git)
    cd stellar-classification
    ```
2.  **Install dependencies** (see below).
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Stellar-Classification.ipynb
    ```
The notebook contains the full workflow, and the trained models are saved to `.joblib` files for future use.

---

## 📦 Dependencies

This project uses standard Python data science libraries. You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib