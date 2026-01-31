# India Road Accident Prediction

## Project Overview
Road accidents are a major public safety concern in India, causing significant loss of life and property each year.  
This project focuses on analyzing historical Indian road accident data and predicting the number of accidents (`Accident_Count`) using machine learning.

A Streamlit-based web application is developed to allow users to upload datasets and generate real-time accident count predictions.


## Objectives
- Analyze Indian road accident data
- Identify accident patterns and trends
- Predict the number of accidents using machine learning
- Build an interactive and user-friendly web application



##  Dataset Description
The dataset contains historical road accident records with features such as:
- Year
- State / Region
- Vehicle Type
- Road Condition
- Weather Condition
- Time-related factors

**Target Variable:** `Accident_Count`


## Machine Learning Approach
- **Problem Type:** Regression
- **Model Used:** Random Forest Regressor
- **Categorical Encoding:** One-Hot Encoding
- **Train-Test Split:** 80% Training, 20% Testing
- **Evaluation Metrics:**
  - R² Score
  - Mean Absolute Error (MAE)


## Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit


## Application Features
- Upload CSV dataset
- Automatic preprocessing of data
- Model training and evaluation
- Interactive input form for prediction
- Real-time accident count prediction

## How to Run the Project

pip install streamlit scikit-learn pandas numpy
streamlit run app.py