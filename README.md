# Heart Disease Prediction Web App

This project is a FastAPI-based web application that predicts the risk of heart disease using a machine learning pipeline.

## Features
- Logistic Regression model with hyperparameter tuning
- Custom preprocessing for selective numeric transformations
- Handles class imbalance using SMOTE
- Interactive HTML UI using FastAPI + Jinja2
- End-to-end ML pipeline deployment

## Tech Stack
- Python
- FastAPI
- scikit-learn
- imbalanced-learn
- Pandas, NumPy
- HTML (Jinja2 templates)

## How to Run Locally

```bash
pip install -r requirements.txt
python -m uvicorn App:app --reload