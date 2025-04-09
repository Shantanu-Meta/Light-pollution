import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

def run_keyfactor_model(target_state):
    # Load dataset
    df = pd.read_csv('All.csv', encoding='latin1')

    # Rename columns
    df.rename(columns={
        'Agriculture_SOL': 'agricultural_SOL',
        'Urban_SOL': 'urban_SOL',
        'Urban_SOL_Change': 'changeOfSOL',
        'Area (km²)': 'area_km2',
        'GSDP(INR billions)': 'gdp'
    }, inplace=True)

    # Clean numeric columns
    for col in ['agricultural_SOL', 'urban_SOL', 'changeOfSOL', 'area_km2', 'gdp']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')

    # Drop rows with missing essential values
    df.dropna(subset=['agricultural_SOL', 'urban_SOL', 'area_km2', 'gdp'], inplace=True)

    # Filter for selected state
    state_data = df[df['state_name'].str.lower() == target_state.strip().lower()]

    if state_data.empty:
        st.warning(f"No data found for '{target_state.title()}'. Please check the spelling and try again.")
        return

    if state_data['area_km2'].isnull().all():
        st.warning(f"⚠️ Warning: 'area_km2' data is missing for {target_state.title()}. Model may be inaccurate.")

    # Define features and target
    X_state = state_data[['agricultural_SOL', 'urban_SOL', 'area_km2']]
    y_state = state_data['gdp']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_state)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_state, test_size=0.3, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Feature Importance
    features = ['agricultural_SOL', 'urban_SOL', 'area_km2']
    importance = model.feature_importances_
    key_factor = features[np.argmax(importance)]
    st.success(f"🌟 Key Factor Influencing Light Pollution in {target_state.title()} : **{key_factor}**")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # Convert to %
    mean_squared_error_val = mean_squared_error(y_test, y_pred)

    # Interpretation Logic
    # def interpret_mae(val):
    #     return "🟢 Low (Good)" if val <= 50 else "🟡 Moderate" if val <= 150 else "🔴 High (Bad)"

    # def interpret_rmse(val):
    #     return "🟢 Low (Good)" if val <= 75 else "🟡 Moderate" if val <= 200 else "🔴 High (Bad)"

    def interpret_r2(val):
        return "🟢 Excellent varient" if val >= 0.65 else "🟡 Moderate varient" if val >= 0.4 else "🔴 Poor"

    def interpret_mape(val):
        return "🟢 Accurate" if val <= 15 else "🟡 Moderately Accurate" if val <= 50 else "🔴 Low Accuracy"

    # Display metrics
    st.subheader(f"📊 Performance Metrics for {target_state.title()}")
    st.markdown(f"""
    - **R² Score**: `{r2:.4f}` → {interpret_r2(r2)}  
      _(Coefficient of Determination - How well the model explains variance in the data)_

    - **MAPE**: `{mape:.2f}%` → {interpret_mape(mape)}  
      _(Mean absolute percent error, lower the value, better the result)_

    """)

    # Residual Plot
    st.subheader("🎆 Residual Analysis")
    st.caption("A well-performing model shows randomly scattered residuals around the zero line (no visible trend).")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred - y_test, alpha=0.6, label="Residuals")
    ax.axhline(0, color="red", linestyle="dashed", linewidth=2, label="Zero Error Line")
    ax.set_xlabel("Actual SOL Values")
    ax.set_ylabel("Residuals (Predicted - Actual)")
    ax.set_title("Residual Plot")
    ax.legend()
    st.pyplot(fig)

    # Add this after st.pyplot(fig)
    residuals = y_pred - y_test
    std_dev = np.std(residuals)

    if abs(np.mean(residuals)) < 0.1 * std_dev and std_dev < 0.3 * np.mean(y_test):
        quality = "🟢 Good Residual Pattern (Randomly scattered)"
    elif std_dev < 0.5 * np.mean(y_test):
        quality = "🟡 Moderate Residual Pattern"
    else:
        quality = "🔴 Poor Residual Pattern (strong pattern or variance)"

    st.info(f"📈 **Residual Quality**: {quality}")
