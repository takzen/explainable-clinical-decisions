# Explainable AI (XAI) for Clinical Decision Support

### An interactive clinical decision support tool that uses a Random Forest model to predict heart disease risk and leverages SHAP (SHapley Additive exPlanations) to provide explainable AI insights.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-orange?logo=streamlit) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange) 
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blue)



## 🚀 Overview

This project addresses a critical challenge in modern AI: the "black box" problem. While complex models like Random Forests can achieve high accuracy, their decision-making processes are often opaque. This application demonstrates a senior-level approach to building not just a predictive model, but an **interpretable and trustworthy AI system** for a high-stakes domain like clinical diagnostics.

The tool predicts the risk of heart disease based on patient data and, most importantly, uses **SHAP (SHapley Additive exPlanations)** to generate a personalized, visual explanation for each prediction. This transforms the model from a black box into a transparent decision-support tool.

## ✨ Key Features & Techniques

*   **Explainable AI (XAI):** The core of the project. It showcases the ability to go beyond prediction and implement state-of-the-art techniques for model interpretability, a key requirement for senior AI/ML roles.
*   **SHAP Integration:** Demonstrates proficiency with the **SHAP library**, the industry standard for explaining the output of any machine learning model. A `TreeExplainer` is trained and serialized alongside the classification model.
*   **Two-Phase Architecture (Offline/Online):**
    1.  **Offline Training (`model_trainer.py`):** A script that trains the `RandomForestClassifier` and the corresponding SHAP explainer, saving them as efficient `.joblib` files.
    2.  **Online Inference (`app.py`):** A **Streamlit** application that loads the pre-trained artifacts to provide real-time, interactive predictions and explanations.
*   **Advanced Data Visualization:** Implements interactive **SHAP force plots**, which provide a clear, intuitive visualization of the factors driving an individual prediction.
*   **Real-World Application:** Applies AI to a meaningful, real-world problem, demonstrating an understanding of how to build responsible and useful AI tools.

## 🛠️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/takzen/explainable-clinical-decisions.git
    cd explainable-clinical-decisions
    ```

2.  **Download the Dataset:**
    *   Download the "Heart Disease UCI" dataset from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).
    *   Create a `data/` folder in the project root.
    *   Place the downloaded `heart.csv` file inside `data/` and rename it to `heart_disease_uci.csv`.

3.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install streamlit pandas scikit-learn shap matplotlib joblib
    ```

4.  **Train the Model and Explainer (Offline Step):**
    *   Run the training script from your terminal.
        ```bash
        python model_trainer.py
        ```
    *   This will create `trained_model.joblib` and `shap_explainer.joblib` in your project root.

5.  **Run the Streamlit Application (Online Step):**
    ```bash
    streamlit run app.py
    ```

## 🖼️ Showcase

| 1. User Inputs Patient Data                             | 2. AI Prediction with SHAP Explanation                  |
| :-------------------------------------------------------- | :------------------------------------------------------ |
| ![User Input](images/01_user_input.png)                   | ![XAI Output](images/02_shap_explanation.png)           |
| *The user adjusts sliders and dropdowns to input anonymous patient data.* | *The app provides a clear prediction and an interactive force plot explaining which factors influenced the decision.* |