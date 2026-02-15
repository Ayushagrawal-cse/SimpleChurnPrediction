📊 Customer Churn Prediction Web App
A machine learning-powered web application that predicts the probability of customer churn using a trained Artificial Neural Network (ANN).

The app is built with TensorFlow, Scikit-learn, and deployed using Streamlit.

🚀 Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their service.

This application:

Accepts customer details through an interactive web interface

Applies preprocessing (scaling + encoding)

Uses a trained ANN model to predict churn probability

Displays whether the customer is likely to churn.

🛠️ Tech Stack

Language: Python 3

Frontend: Streamlit

Machine Learning: TensorFlow (Keras)

Data Processing: Pandas, NumPy

Preprocessing: Scikit-learn (StandardScaler, OneHotEncoder, LabelEncoder)

Model Type: Artificial Neural Network (ANN)

📂 Project Structure
├── app.py
├── model.h5
├── scaler.pkl
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
└── README.md

🧠 Model Details

Built using TensorFlow Sequential API

Fully connected dense layers

Binary classification output (Sigmoid activation)

Loss Function: Binary Crossentropy

Optimizer: Adam

📈 Features Used for Prediction
Credit Score

Geography (One-Hot Encoded)

Gender (Label Encoded)

Age

Tenure

Balance

Number of Products

Has Credit Card

Is Active Member

Estimated Salary

📊 Output

The model returns:

  Churn Probability (0 to 1)
  
Final classification:

  Likely to churn
  
  Not likely to churn

🎯 Key Learning Outcomes

End-to-end ML workflow implementation

Data preprocessing and feature engineering

Model serialization and loading

Deploying deep learning models with Streamlit

Real-time prediction interface

🚀 Future Improvements
Add model performance metrics (Accuracy, ROC-AUC, Confusion Matrix)

Add feature importance visualization

Deploy using Docker

Convert model to TensorFlow Lite for lightweight deployment

Add batch CSV upload for bulk predictions
Integrate database for storing prediction logs
Improve UI with custom Streamlit themes
