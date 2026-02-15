# Steps to Deploy the Streamlit App

Here are the steps to deploy your Streamlit app to Streamlit Community Cloud:

### 1. **Prepare Your `requirements.txt`**

Ensure your `requirements.txt` file accurately lists all the Python libraries your app needs to run. Based on your project, the file should look something like this:

```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
xgboost
joblib
```

You can generate this file by running `pip freeze > requirements.txt` in your terminal, but it's good practice to manually review it to include only necessary dependencies.

### 2. **Push Your Project to GitHub**

Streamlit Community Cloud deploys apps directly from a GitHub repository. Your repository must contain the following:

*   `app.py` (your main Streamlit application script)
*   `requirements.txt` (the list of dependencies)
*   The `model/` directory containing all your `.pkl` model and scaler files.
*   Any other files your app needs to run.

### 3. **Sign Up for Streamlit Community Cloud**

1.  Go to the [Streamlit Community Cloud](https://share.streamlit.io) website.
2.  Click "Sign up" and choose to continue with your GitHub account. You will need to authorize Streamlit to access your repositories.

### 4. **Deploy the App**

1.  From your Streamlit Community Cloud dashboard, click the "**New app**" button in the top-right corner.
2.  **Choose a repository**: Select the GitHub repository where you pushed your project.
3.  **Select a branch**: Choose the branch that contains the code you want to deploy (e.g., `main` or `master`).
4.  **Set the main file path**: Ensure this is set to your main script, which is `app.py`.
5.  Click the "**Deploy!**" button.

Streamlit will now build your application by installing the dependencies from `requirements.txt` and then run your `app.py` script. You will be redirected to the URL of your live application once the deployment is complete.
