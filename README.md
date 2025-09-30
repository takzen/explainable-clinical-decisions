# Explainable AI (XAI) for Clinical Decision Support

### An interactive clinical decision support tool that uses an optimized Random Forest model to predict heart disease risk and leverages SHAP (SHapley Additive exPlanations) to provide explainable AI insights.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange?logo=streamlit) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange) 
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange?logo=jupyter)


## 🚀 Overview

This project addresses a critical challenge in modern AI: the "black box" problem. While complex models like Random Forests can achieve high accuracy, their decision-making processes are often opaque. This application demonstrates a senior-level approach to building not just a predictive model, but an **interpretable and trustworthy AI system** for a high-stakes domain like clinical diagnostics.

The tool predicts the risk of heart disease based on patient data and, most importantly, uses **SHAP (SHapley Additive exPlanations)** to generate a personalized, visual explanation for each prediction. The entire analytical workflow, from data cleaning and exploration to hyperparameter tuning, is documented in a comprehensive Jupyter Notebook.

## ✨ Key Features & Techniques

*   **End-to-End ML Workflow:** The project includes a detailed Jupyter Notebook (`Model_Training_Analysis.ipynb`) that covers the entire data science process: **EDA, data cleaning, baseline model evaluation, hyperparameter tuning with GridSearchCV, and final model validation.**
*   **Explainable AI (XAI):** The core of the project. It showcases the ability to go beyond prediction and implement state-of-the-art techniques for model interpretability, a key requirement for senior AI/ML roles.
*   **Hyperparameter Optimization:** Demonstrates proficiency with `GridSearchCV` to systematically find the optimal parameters for the `RandomForestClassifier`, resulting in a more robust and generalizable model.
*   **SHAP Integration:** Shows expertise with the **SHAP library**, the industry standard for explaining model outputs. A `TreeExplainer` is trained on the final, optimized model and serialized for production use.
*   **Two-Phase Architecture (Offline/Online):**
    1.  **Offline Analysis & Training (`Model_Training_Analysis.ipynb`):** A notebook that serves as the single source of truth for all analytical work, producing the final, optimized model and SHAP explainer.
    2.  **Online Inference (`app.py`):** A **Streamlit** application that loads the pre-trained artifacts to provide real-time, interactive predictions and explanations.

## 🛠️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/takzen/explainable-clinical-decisions.git
    cd explainable-clinical-decisions
    ```

2.  **Download the Dataset:**
    *   Download the "Heart Disease UCI" dataset from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).
    *   Create a `data/` folder in the project root.
    *   Place the downloaded `heart_disease_uci.csv` file inside `data/`.

3.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

4.  **Explore the Analysis & Train the Model (Offline Step):**
    *   To understand the full data science process, open the Jupyter Notebook:
        ```bash
        jupyter lab notebooks/Model_Training_Analysis.ipynb
        ```
    *   Running this notebook will generate the final `trained_model.joblib` and `shap_explainer.joblib` files in the root directory.

5.  **Run the Streamlit Application (Online Step):**
    ```bash
    streamlit run app.py
    ```

## 🖼️ Showcase

| 1. User Inputs Patient Data                             | 2. AI Prediction with SHAP Explanation                  |
| :-------------------------------------------------------- | :------------------------------------------------------ |
| ![User Input](images/01_user_input.png)                   | ![XAI Output](images/02_shap_explanation.png)           |
| *The user adjusts sliders and dropdowns to input anonymous patient data.* | *The app provides a clear prediction and a SHAP plot explaining which factors influenced the decision.* |