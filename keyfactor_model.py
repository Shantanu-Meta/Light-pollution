import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, explained_variance_score)

def run_keyfactor_model(target_state):
    # Load dataset
    df = pd.read_csv('All.csv', encoding='latin1')

    # Rename columns for consistency
    df.rename(columns={
        'Agriculture_SOL': 'agricultural_SOL',
        'Urban_SOL': 'urban_SOL',
        'Urban_SOL_Change': 'changeOfSOL',
        'Area (km²)': 'area_km2',
        'GSDP(INR billions)': 'gdp'
    }, inplace=True)

    # Convert string numbers to float, ensuring proper cleaning
    for col in ['agricultural_SOL', 'urban_SOL', 'changeOfSOL', 'area_km2', 'gdp']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')

    # Drop rows where essential values are missing
    df.dropna(subset=['agricultural_SOL', 'urban_SOL', 'area_km2', 'gdp'], inplace=True)

    # Filter data for the given state
    state_data = df[df['state_name'].str.lower() == target_state.strip().lower()]

    if state_data.empty:
        st.warning(f"No data found for '{target_state.title()}'. Please check the spelling and try again.")
        return

    # Check if `area_km2` is missing
    if state_data['area_km2'].isnull().all():
        st.warning(f"⚠️ Warning: 'area_km2' data is missing for {target_state.title()}. Model may be inaccurate.")

    # Features and target
    X_state = state_data[['agricultural_SOL', 'urban_SOL', 'area_km2']]
    y_state = state_data['gdp']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_state)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_state, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Feature importance
    features = ['agricultural_SOL', 'urban_SOL', 'area_km2']
    importance = model.feature_importances_
    key_factor = features[np.argmax(importance)]

    # Display key factor
    st.success(f"Key Factor Influencing Light Pollution in {target_state.title()}:  **{key_factor}**")

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Display performance metrics with captions
    st.subheader(f"📊 Performance Metrics for {target_state.title()}")
    st.markdown(f"""
    - **Mean Absolute Error (MAE)**: {mae:.2f} 🔍 _(Lower is better, represents average absolute error)_
    - **Root Mean Squared Error (RMSE)**: {rmse:.2f} 🎯 _(Gives an idea of the model's average error size)_
    - **R² Score**: {r2:.4f} 📈 _(Closer to 1 is better, shows how well the model explains variance)_
    """)

   # Residual plot to analyze model performance visually
    st.subheader("🎆 Residual Analysis")
    st.caption("""
    This indicates that the model’s errors are unbiased and there is no pattern (underfitting/overfitting).
    """)

    # Create residual plot without requiring statsmodels
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred - y_test, alpha=0.6, label="Residuals")
    ax.axhline(0, color="red", linestyle="dashed", linewidth=2, label="Zero Error Line")
    ax.set_xlabel("Actual key Values")
    ax.set_ylabel("Residual (Predicted - Actual)")
    ax.set_title("Residual Plot")
    ax.legend()
    st.pyplot(fig)
