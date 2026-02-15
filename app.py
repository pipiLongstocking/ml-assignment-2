
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


# Load the original dataset to use as default test data
@st.cache_data
def load_default_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    data['quality'] = data['quality'].apply(lambda q: 'good' if q >= 7 else 'bad')
    data['quality'] = data['quality'].map({'good': 1, 'bad': 0})
    X = data.drop('quality', axis=1)
    y = data['quality']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test

X_test_default, y_test_default = load_default_data()


# Load models and scaler
@st.cache_resource
def load_models():
    models = {}
    model_files = ["Logistic_Regression.pkl", "Decision_Tree.pkl", "kNN.pkl", "Naive_Bayes.pkl", "Random_Forest.pkl", "XGBoost.pkl"]
    for file in model_files:
        model_name = file.replace("_", " ").replace(".pkl", "")
        models[model_name] = joblib.load(f"model/{file}")
    scaler = joblib.load("model/scaler.pkl")
    return models, scaler

models, scaler = load_models()

st.title("Wine Quality Classification App")
st.write("This app classifies the quality of red wine using various machine learning models.")

# File uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    # Assume the uploaded file has the same columns as the training data, except for the target variable 'quality'
    if 'quality' in input_df.columns:
        X_test = input_df.drop('quality', axis=1)
        y_test = input_df['quality']
        st.write("Using uploaded data for evaluation.")
    else:
        X_test = input_df
        y_test = None # No ground truth for evaluation
        st.write("Using uploaded data for prediction (no evaluation).")

else:
    st.write("No file uploaded. Using the default test dataset for demonstration.")
    X_test = X_test_default
    y_test = y_test_default


# Model selection
model_name = st.selectbox("Select a model", list(models.keys()))

# Prediction and evaluation
if st.button("Predict and Evaluate"):
    if X_test is not None:
        # Scale the features
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        model = models[model_name]
        y_pred = model.predict(X_test_scaled)

        st.subheader(f"Results for {model_name}")

        if y_test is not None:
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))

            # Evaluation metrics
            st.write("### Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
            col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            col1.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            col2.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
            col3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        else:
            st.write("### Predictions")
            predictions_df = pd.DataFrame(X_test)
            predictions_df['Predicted Quality'] = ['Good' if p == 1 else 'Bad' for p in y_pred]
            st.dataframe(predictions_df)


st.sidebar.header("About")
st.sidebar.info(
    "This is a demo application to showcase different classification models "
    "for the Wine Quality dataset. The models are trained on the red wine quality dataset from the UCI Machine Learning Repository."
)

st.sidebar.header("Models")
st.sidebar.info(
    "The following models are available:
"
    "- Logistic Regression
"
    "- Decision Tree
"
    "- k-Nearest Neighbors (kNN)
"
    "- Naive Bayes
"
    "- Random Forest
"
    "- XGBoost"
)
