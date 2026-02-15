# Wine Quality Classification

This project implements and evaluates six different machine learning models to classify the quality of red wine. It also includes an interactive web application built with Streamlit to demonstrate the models' performance.

## Problem Statement

The goal of this project is to build a classification system that can predict whether a red wine is of "good" or "bad" quality based on its physicochemical properties. This is a binary classification problem where "good" quality is defined as a wine with a quality score of 7 or higher, and "bad" quality is a wine with a score lower than 7.

## Dataset Description

The dataset used is the **Red Wine Quality** dataset from the UCI Machine Learning Repository. It contains 1,599 instances of red wine samples, each with 11 physicochemical features and a quality score ranging from 3 to 8.

**Features:**
1.  fixed acidity
2.  volatile acidity
3.  citric acid
4.  residual sugar
5.  chlorides
6.  free sulfur dioxide
7.  total sulfur dioxide
8.  density
9.  pH
10. sulphates
11. alcohol

**Target Variable:**
- `quality` (score from 0 to 10) - transformed into a binary variable (1 for good, 0 for bad).

## Models Used

Six different classification models were trained and evaluated. The performance of each model on the test set is summarized in the table below.

### Model Performance Comparison

| ML Model Name     | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression| 0.8938   | 0.8804 | 0.6957    | 0.3721 | 0.4848 | 0.4580 |
| Decision Tree     | 0.9062   | 0.8182 | 0.6383    | 0.6977 | 0.6667 | 0.6131 |
| kNN               | 0.8938   | 0.8237 | 0.6667    | 0.4186 | 0.5143 | 0.4738 |
| Naive Bayes       | 0.8594   | 0.8517 | 0.4844    | 0.7209 | 0.5794 | 0.5131 |
| Random Forest     | 0.9375   | 0.9546 | 0.9259    | 0.5814 | 0.7143 | 0.7045 |
| XGBoost           | 0.9406   | 0.9422 | 0.8750    | 0.6512 | 0.7467 | 0.7239 |

### Observations on Model Performance

| ML Model Name     | Observation about model performance                                                                                                                              |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression| Provides a decent baseline with good accuracy and AUC, but struggles with precision and recall, indicating difficulty in correctly identifying "good" quality wines.      |
| Decision Tree     | Shows a good balance between precision and recall, suggesting it can identify both classes reasonably well. It has a higher F1 score than logistic regression.          |
| kNN               | Performs similarly to Logistic Regression in terms of accuracy, but with slightly better precision and recall. Still, it is not as balanced as the Decision Tree.          |
| Naive Bayes       | Exhibits high recall but very low precision, meaning it identifies a high proportion of "good" wines but also misclassifies many "bad" wines as "good".                    |
| Random Forest     | Achieves high accuracy, AUC, and precision. It is one of the top-performing models, indicating that the ensemble approach is effective for this dataset.                |
| XGBoost           | The best-performing model across most metrics, with the highest F1 and MCC scores. It provides a great balance of precision and recall and is very accurate. |

## How to Run the Streamlit App

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

Then open your browser to `http://localhost:8501`.
