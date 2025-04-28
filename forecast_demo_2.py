import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import datetime
import pmdarima

# Load data & models
test_df = pd.read_csv("test_power_consumption_2.csv", parse_dates=['dt'])
X_test_rf = np.load("test_power_consumption_rf_2.npy")
X_test_lstm = np.load("test_power_consumption_lstm_2.npy")

arma_model = joblib.load("arma_model_2.pkl")
rf_model = joblib.load("rf_model_2.pkl")
lstm_model = load_model("lstm_model_2.keras", compile=False)

# Load scaler
scaler = joblib.load("scaler_2.pkl")

# Forecasts (full length)
arma_preds = arma_model.forecast(steps=len(test_df))
rf_preds = rf_model.predict(X_test_rf)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()
lstm_preds_inversed = scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()

# App UI
st.title("Pronóstico de Ventas Minoristas")

model_choice = st.selectbox("Elegir un Modelo", ["ARMA", "Random Forest", "LSTM"])
forecast_type = st.radio("Forecast type", ["Pronóstico Puntual (por día)", "Pronóstico Acumulado"])

st.subheader("Seleccionar fecha inicial y final (con fecha completa)")

# Get min and max datetime from the dataset
min_dt = test_df['dt'].min()
max_dt = test_df['dt'].max()

# Select date
start_date = st.date_input("Fecha Inicial", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
end_date = st.date_input("Fecha Final", value=max_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())

# Filter dataframe
mask = (test_df['dt'] >= start_date) & (test_df['dt'] <= end_date)
filtered_df = test_df[mask]

# Prevent empty selections
if filtered_df.empty:
    st.warning("No data available in the selected datetime range.")
    st.stop()

# Get position indices for slicing predictions
start_pos = test_df.index.get_loc(filtered_df.index[0])
end_pos = test_df.index.get_loc(filtered_df.index[-1]) + 1

# Slice forecasts
if model_choice == "ARMA":
    preds = arma_preds[start_pos:end_pos]
elif model_choice == "Random Forest":
    preds = rf_preds[start_pos:end_pos]
elif model_choice == "LSTM":
    preds = lstm_preds_inversed[start_pos:end_pos]

# Get actual values
actual = test_df['Global_active_power'].iloc[start_pos:end_pos].values

# Apply cumulative forecast
if forecast_type == "Pronóstico Acumulado":
    preds = np.cumsum(preds) + test_df['Global_active_power'].iloc[start_pos]

# Prepare chart dataframe
n = min(len(filtered_df), len(preds), len(actual))
chart_df = pd.DataFrame({
    'Datetime': filtered_df['dt'].values[:n],
    'Actual': actual[:n],
    'Forecast': preds[:n]
})

# Altair line chart with y-axis label
chart = alt.Chart(chart_df).transform_fold(
    ['Actual', 'Forecast'],
    as_=['Series', 'Value']
).mark_line().encode(
    x='Datetime:T',
    y=alt.Y('Value:Q', title='Power Consumption (kWh)'),
    color='Series:N'
).properties(
    width=800,
    height=400
)

st.altair_chart(chart, use_container_width=True)
