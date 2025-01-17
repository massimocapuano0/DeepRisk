# DISCLAIMER:
# This code is provided for educational purposes only and is not intended as financial advice.
# Use at your own risk, and consult a financial professional for investment decisions.

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
import datetime

yf.pdr_override()
plt.style.use('ggplot') 


def download_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        data.dropna(inplace=True) 
        return data
    except Exception as e:
        print(f"Error while downloading data: {e}")
        return None

def prepare_data(series, time_step=60):
    data = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    x, y = [], []
    for i in range(time_step, len(data_scaled)):
        x.append(data_scaled[i-time_step:i, 0])
        y.append(data_scaled[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(50, return_sequences=False),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def predict_future_consistent(model, data, scaler, n_days, n_repeats=5):
    last_data = data[-1].reshape(1, -1, 1)
    all_predictions = []

    for _ in range(n_repeats):
        predictions = []
        temp_last_data = last_data.copy()
        for _ in range(n_days):
            prediction = model.predict(temp_last_data)
            predictions.append(prediction[0, 0])
            temp_last_data = np.append(temp_last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
        all_predictions.append(predictions)

    mean_predictions = np.mean(all_predictions, axis=0)
    mean_predictions = scaler.inverse_transform(np.array(mean_predictions).reshape(-1, 1))
    return mean_predictions

def historical_var(returns, confidence=0.05):
    return np.percentile(returns, confidence * 100)

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def expected_shortfall(returns, var):
    return returns[returns <= var].mean()

def parametric_var(mean, std, confidence=0.05):
    z = np.abs(np.percentile(np.random.normal(0, 1, 100000), confidence * 100))
    return mean - z * std

def monte_carlo_var(returns, confidence=0.05, simulations=100000):
    mean = np.mean(returns)
    std = np.std(returns)
    simulated_returns = np.random.normal(mean, std, simulations)
    return np.percentile(simulated_returns, confidence * 100), simulated_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe_ratio

start_date = "2012-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
tickers = ['GLD', 'SLV', 'USO']
weights = [0.4, 0.4, 0.2] 

data = download_data(tickers, start_date, end_date)
if data is None:
    exit()

data['Portfolio'] = (data * weights).sum(axis=1)

plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(data[ticker], label=ticker)
plt.plot(data['Portfolio'], label='Portfolio', linewidth=2, linestyle='--')
plt.title('Performance')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

returns = data['Portfolio'].pct_change().dropna()
x, y, scaler = prepare_data(data['Portfolio'])

train_size = int(len(x) * 0.8)
test_size = len(x) - train_size
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

model = build_lstm((x_train.shape[1], 1))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])

def predict_lstm(model, x_data, scaler):
    predictions = model.predict(x_data)
    return scaler.inverse_transform(predictions)

train_predictions = predict_lstm(model, x_train, scaler)
test_predictions = predict_lstm(model, x_test, scaler)
test_mae = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predictions)

train_rmse = calculate_rmse(scaler.inverse_transform(y_train.reshape(-1, 1)), train_predictions)
test_rmse = calculate_rmse(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predictions)

predicted_returns = (test_predictions[1:] - test_predictions[:-1]) / test_predictions[:-1]
actual_returns = returns[-len(predicted_returns):].values

predicted_returns = np.clip(predicted_returns, -0.1, 0.1)
actual_returns = np.clip(actual_returns, -0.1, 0.1)

var_historical = historical_var(actual_returns)
residuals = predicted_returns.flatten() - actual_returns
var_lstm = historical_var(residuals)

plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(test_predictions):], scaler.inverse_transform(y_test.reshape(-1, 1)), label="Real")
plt.plot(data.index[-len(test_predictions):], test_predictions, label="LSTM")
plt.title("Comparison real vs LSTM analysis ")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss (Train)')
plt.plot(history.history['val_loss'], label='Loss (Validation)')
plt.title('Loss during training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

confidence_level = 0.05 
mean_return = np.mean(actual_returns)
std_return = np.std(actual_returns)

var_historical = historical_var(actual_returns, confidence=confidence_level)
es_historical = expected_shortfall(actual_returns, var_historical)

var_parametric = parametric_var(mean_return, std_return, confidence=confidence_level)
es_parametric = mean_return - (std_return * np.abs(np.percentile(np.random.normal(0, 1, 100000), (confidence_level / 2) * 100)))

var_monte_carlo, simulated_returns = monte_carlo_var(actual_returns, confidence=confidence_level)
es_monte_carlo = expected_shortfall(simulated_returns, var_monte_carlo)

results = pd.DataFrame({
    "Method": ["Historical", "Parametric", "Monte Carlo"],
    "VaR (95%)": [var_historical, var_parametric, var_monte_carlo],
    "Expected Shortfall (95%)": [es_historical, es_parametric, es_monte_carlo]
})

plt.figure(figsize=(14, 8))
plt.hist(actual_returns, bins=50, alpha=0.5, label='Actual Returns', color='blue', density=True)
plt.axvline(var_historical, color='red', linestyle='--', label='Historical VaR')
plt.axvline(var_parametric, color='green', linestyle='--', label='Parametric VaR')
plt.axvline(var_monte_carlo, color='orange', linestyle='--', label='Monte Carlo VaR')
plt.axvline(var_lstm, color='purple', linestyle='--', label='LSTM VaR')
plt.title('Comparison of VaR Calculation Methods')
plt.xlabel('Returns')
plt.ylabel('Normalized Frequency')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 8))
plt.hist(simulated_returns, bins=50, alpha=0.5, color='purple', density=True, label='Simulated Returns (Monte Carlo)')
plt.axvline(var_monte_carlo, color='orange', linestyle='--', label='Monte Carlo VaR')
plt.axvline(es_monte_carlo, color='darkorange', linestyle='--', label='Monte Carlo ES')
plt.axvline(var_lstm, color='purple', linestyle='--', label='LSTM VaR')
plt.title('Monte Carlo Simulated Distribution-LSTM VaR Comparison')
plt.xlabel('Simulated Returns')
plt.ylabel('Normalized Frequency')
plt.legend()
plt.grid(True)
plt.show()

print(f"""
--- Results ---
Train set: {train_size} samples
Test set: {test_size} samples
RMSE Train: {train_rmse:.4f}, RMSE Test: {test_rmse:.4f}
MAE Test: {test_mae:.4f}
VaR Historical: {var_historical:.4f}, VaR LSTM: {var_lstm:.4f}

Comparison of VaR and Expected Shortfall:
{results}
""")
