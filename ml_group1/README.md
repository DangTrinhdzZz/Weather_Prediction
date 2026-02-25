"# ml-group1" 
# 🌤️ Hanoi Temperature Forecasting Project

This project aims to **predict daily (and hourly) temperatures in Hanoi** using various Machine Learning models.  
It is developed as part of the **Machine Learning I course**, with the goal of building an end-to-end ML product — from data collection to model deployment with a UI demo.

---

## 🚀 Project Overview

The project follows a typical machine learning pipeline:

1. **Data Collection:**  
   Collect 10 years of Hanoi weather data from [Visual Crossing Weather API](https://www.visualcrossing.com/weather-query-builder/Hanoi/us/last15days/).

2. **Data Understanding:**  
   Explore and explain 33 weather-related features (e.g., `temperature`, `humidity`, `moonphase`, etc.), and visualize trends of Hanoi temperature over time.

3. **Data Processing:**  
   Handle missing values, normalize features, encode categorical variables, and compute correlations.

4. **Feature Engineering:**  
   Create new features that improve prediction accuracy — e.g., rolling averages, lag features, or text-based weather descriptions.

5. **Model Training & Evaluation:**  
   Train and tune models (Random Forest, XGBoost, LSTM, etc.) and evaluate with RMSE, R², and MAPE.  
   Optionally, use **Optuna** for hyperparameter tuning and **ClearML** for experiment tracking.

6. **Deployment:**  
   Build an interactive demo using **Streamlit** or **Gradio** to visualize model predictions.

7. **ONNX Conversion (Optional):**  
   Convert trained models to ONNX format for efficient deployment and inference.

---

## 🧱 Project Structure

```bash
group1/
├─ data/
│  ├─ hanoi_weather.csv          # Raw dataset (downloaded from Visual Crossing)
│  └─ processed.csv              # Cleaned dataset (after preprocessing, optional)
│
├─ src/
│  ├─ data_preprocessing.py      # Load, clean data, handle missing values
│  ├─ feature_engineering.py     # Create new features for better prediction
│  ├─ model_training.py          # Train ML model, save it to /models
│  ├─ model_evaluation.py        # Evaluate model (RMSE, R², etc.)
│  └─ app.py                     # Streamlit / Gradio app for demo UI
│
├─ models/
│  └─ model.pkl                  # Trained model file (auto-created after training)
│
|─ Data_understanding            # Step 2: phân tích data, correlation matrix , ... vv 
├─ requirements.txt              # List of Python dependencies
├─ main.py                       # Main script to run full pipeline
├─ .gitignore                    # Ignore unnecessary files (venv, data/raw, etc.)
└─ README.md                     # Project documentation
