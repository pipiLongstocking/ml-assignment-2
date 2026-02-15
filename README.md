# Wine Quality Classification

## Problem Statement

This project aims to classify the quality of red wine as 'good' or 'bad' based on its physicochemical properties. This is a binary classification problem.

## Dataset Description

The dataset used is the Wine Quality dataset from the UCI Machine Learning Repository. It contains 1599 instances of red wine, each with 11 physicochemical properties (e.g., fixed acidity, volatile acidity, citric acid, etc.) and a quality score from 0 to 10. For this project, the quality score is converted to a binary variable where wines with a score of 7 or higher are considered 'good' and the rest are 'bad'.

## Models Used

The following classification models were implemented:

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbor Classifier
4.  Naive Bayes Classifier
5.  Random Forest
6.  XGBoost

### Model Performance Comparison

**Note to user:** Please run the Streamlit app, select each model, and fill in the following table with the metrics displayed in the app.

| ML Model Name     | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ----------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8938   | 0.8804 | 0.6957    | 0.3721 | 0.4848   | 0.4580 |
| Decision Tree     | 0.9062   | 0.8182 | 0.6383    | 0.6977 | 0.6667   | 0.6131 |
| kNN               | 0.8938   | 0.8237 | 0.6667    | 0.4186 | 0.5143   | 0.4738 |
| Naive Bayes       | 0.8594   | 0.8517 | 0.4844    | 0.7209 | 0.5794   | 0.5131 |
| Random Forest     | 0.9375   | 0.9546 | 0.9259    | 0.5814 | 0.7143   | 0.7045 |
| XGBoost           | 0.9406   | 0.9422 | 0.8750    | 0.6512 | 0.7467   | 0.7239 |

### Observations on Model Performance

**Note to user:** Please add your observations on the performance of each model on the chosen dataset in the table below.

| ML Model Name           | Observation about model performance                                                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression     | Performs reasonably well with an accuracy of 89.38%, but has a very low recall of 37.21%, indicating it struggles to identify the 'good' quality wines.                 |
| Decision Tree           | Shows a significant improvement in recall (69.77%) over Logistic Regression and kNN, and maintains good accuracy. It provides a balanced performance.                  |
| kNN                     | Similar to Logistic Regression, it has decent accuracy but a low recall (41.86%). It is not as effective as other models in identifying positive class instances.    |
| Naive Bayes             | Achieves the highest recall (72.09%) among all models, but at the cost of very low precision (48.44%). This means it identifies most of the 'good' wines but also incorrectly classifies many 'bad' wines as 'good'. |
| Random Forest (Ensemble) | One of the top performers with high accuracy (93.75%) and the highest precision (92.59%). Its F1 score is also quite high. It's a very reliable model for this task. |
| XGBoost (Ensemble)      | The best performing model overall, with the highest Accuracy, F1 Score, and MCC. It provides a great balance between precision and recall.                            |