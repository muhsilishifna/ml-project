import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="India Road Accident Prediction", layout="wide")

st.title("🚦 India Road Accident Prediction")
st.write("Predict **Accident_Count** using historical Indian road accident data")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Target Column
    # -------------------------------
    TARGET = "Accident_Count"

    if TARGET not in df.columns:
        st.error("❌ 'Accident_Count' column not found in dataset")
        st.stop()

    # -------------------------------
    # Separate X and y
    # -------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Identify categorical & numerical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    # -------------------------------
    # Preprocessing
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    # -------------------------------
    # Model
    # -------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # -------------------------------
    # Model Evaluation
    # -------------------------------
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")

    # -------------------------------
    # Prediction Section
    # -------------------------------
    st.subheader("🔮 Predict Accident Count")

    input_data = {}

    for col in X.columns:
        if col in cat_cols:
            input_data[col] = st.selectbox(
                f"{col}",
                df[col].unique()
            )
        else:
            input_data[col] = st.number_input(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

    input_df = pd.DataFrame([input_data])

    if st.button("🚨 Predict Accident Count"):
        prediction = pipeline.predict(input_df)[0]
        st.success(f"### 🧮 Predicted Accident Count: **{int(prediction)}**")

else:
    st.info("👆 Upload your Indian road accident CSV file to begin")
