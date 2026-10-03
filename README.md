# 🏠 House Price Prediction

A Machine Learning based web application that predicts the estimated sale price of a house based on its characteristics.

The project compares multiple regression algorithms and deploys the best-performing model using Streamlit.

---

## 📌 Project Overview

The goal of this project is to build a regression model capable of predicting house prices using property-related features.

The project follows a complete Machine Learning workflow:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Missing value analysis
- Feature preprocessing
- Train-test split
- Model training
- Model comparison
- Model evaluation
- Model serialization using Joblib
- Interactive Streamlit application

---

## 🤖 Machine Learning Models

The following regression algorithms were evaluated:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 17,934 | 30,240 | 0.779 |
| Decision Tree | 29,443 | 45,604 | 0.498 |
| Random Forest | 16,423 | 29,254 | **0.794** |
| Gradient Boosting | **15,986** | 32,029 | 0.753 |

Random Forest was selected for the Streamlit application based on the evaluation results used in this project.

> Note: The model comparison values are specific to the train/test split used in this project.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 📊 Dataset

The project uses the **House Prices dataset**, containing information about residential properties and their sale prices.

### Target Variable

```text
SalePrice