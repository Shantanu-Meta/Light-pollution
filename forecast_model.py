import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


def forecast_sol(state_data, time_steps=5, epochs=100, batch_size=32):
    """
    Forecasts Urban_SOL and Agriculture_SOL for a given state using LSTM.

    Returns:
        forecast_dict: Forecasted values for 2025 and 2026
        train_mse: MSE on training data
        test_mse: MSE on test data
    """
    # Preprocess data
    state_data['Urban_SOL'] = pd.to_numeric(state_data['Urban_SOL'].str.replace(',', ''), errors='coerce')
    state_data['Agriculture_SOL'] = pd.to_numeric(state_data['Agriculture_SOL'].str.replace(',', ''), errors='coerce')
    state_data = state_data.dropna()

    # Select relevant columns
    data = state_data[['Urban_SOL', 'Agriculture_SOL']].values

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences
    X, Y = [], []
    for i in range(len(scaled_data) - time_steps - 1):
        X.append(scaled_data[i:(i + time_steps), :])
        Y.append(scaled_data[i + time_steps, :])
    X, Y = np.array(X), np.array(Y)

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Build model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=2)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=0)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_mse = mean_squared_error(Y_train, train_preds)
    test_mse = mean_squared_error(Y_test, test_preds)

    # Forecast for 2025 and 2026
    last_sequence = scaled_data[-time_steps:]
    forecasts = []
    for _ in range(5):  # For 2025 and 2026
        x_input = last_sequence.reshape((1, time_steps, 2))
        yhat = model.predict(x_input, verbose=0)
        forecasts.append(scaler.inverse_transform(yhat)[0])
        last_sequence = np.append(last_sequence[1:], yhat, axis=0)

    forecast_dict = {
    "2022": {"Urban_SOL": forecasts[0][0], "Agriculture_SOL": forecasts[0][1]},
    "2023": {"Urban_SOL": forecasts[1][0], "Agriculture_SOL": forecasts[1][1]},
    "2024": {"Urban_SOL": forecasts[2][0], "Agriculture_SOL": forecasts[2][1]},
    "2025": {"Urban_SOL": forecasts[3][0], "Agriculture_SOL": forecasts[3][1]},
    "2026": {"Urban_SOL": forecasts[4][0], "Agriculture_SOL": forecasts[4][1]},
    }


    return forecast_dict, train_mse, test_mse, data  # Return raw data for trend analysis


def run_forecast_model(target_state):
    df = pd.read_csv('All.csv', encoding='latin1')
    df['system:time_start'] = pd.to_datetime(df['system:time_start'])
    df.set_index('system:time_start', inplace=True)

    state_data = df[df['state_name'].str.lower() == target_state.strip().lower()]

    if not state_data.empty:
        forecasts, train_mse, test_mse, historical_data = forecast_sol(state_data)

        # Display forecast
        st.success("✅ Forecasting completed successfully!")
        st.subheader(f"Forecast for {target_state.title()}")
        st.write(f"**2025**: Urban SOL = {forecasts['2025']['Urban_SOL']:.2f}, Agriculture SOL = {forecasts['2025']['Agriculture_SOL']:.2f}")
        st.write(f"**2026**: Urban SOL = {forecasts['2026']['Urban_SOL']:.2f}, Agriculture SOL = {forecasts['2026']['Agriculture_SOL']:.2f}")
        st.caption("ℹ️ In nW/cm²/sr, higher SOL (Sum of Lights) indicate more light pollution.")

        # Model Evaluation
        st.subheader("📊 Model Evaluation")
        st.write(f"**Training MSE**: {train_mse:.6f} {'✅ (Good)' if train_mse < 0.3 else '⚠️ (Can Improve)'}")
        st.write(f"**Testing MSE**: {test_mse:.6f} {'✅ (Good)' if test_mse < 0.3 else '⚠️ (Can Improve)'}")
        st.caption("ℹ️ Lower MSE (Mean Squared Error) means better prediction accuracy.")

        # Trend Analysis
        last_actual = historical_data[-1]
        sol_trend_urban = "📈 Increasing" if forecasts["2026"]["Urban_SOL"] > last_actual[0] else "📉 Decreasing"
        sol_trend_agri = "📈 Increasing" if forecasts["2026"]["Agriculture_SOL"] > last_actual[1] else "📉 Decreasing"

        st.subheader("📈 Trend Analysis (Compared to Latest Available Data)")
        st.write(f"Urban_SOL: {sol_trend_urban}")
        st.write(f"Agriculture_SOL: {sol_trend_agri}")

        # Graph: Time series 2013–2026
        st.subheader("📅 Urban & Agriculture SOL Over Time")

        # Extract and clean original time-series
        state_data['Urban_SOL'] = pd.to_numeric(state_data['Urban_SOL'].astype(str).str.replace(',', ''), errors='coerce')
        state_data['Agriculture_SOL'] = pd.to_numeric(state_data['Agriculture_SOL'].astype(str).str.replace(',', ''), errors='coerce')


        state_data = state_data.dropna()

        # Prepare data for plotting
        plot_df = state_data[['Urban_SOL', 'Agriculture_SOL']].copy()
        last_date = plot_df.index[-1]
        for year in range(2022, 2027):
            plot_df.loc[pd.to_datetime(f"{year}-01-01")] = [
                forecasts[str(year)]["Urban_SOL"],
                forecasts[str(year)]["Agriculture_SOL"]
            ]
        plot_df = plot_df.sort_index()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(plot_df.index, plot_df['Urban_SOL'], marker='o', label='Urban_SOL', color='tab:blue')
        ax.plot(plot_df.index, plot_df['Agriculture_SOL'], marker='o', label='Agriculture_SOL', color='tab:green')
        ax.set_title(f'SOL Trend for {target_state.title()} (2013–2026)')
        ax.set_xlabel('Year')
        ax.set_ylabel('SOL Value')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.error(f"❌ No data found for state: {target_state}")
