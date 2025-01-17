# DeepRisk - ETF Portfolio Forecasting with LSTM Neural Networks

## Overview
This Python script builds a forecasting model for an ETF portfolio using **LSTM (Long Short-Term Memory)** neural networks. The model also calculates **Value at Risk (VaR)** to assess potential portfolio risks. Key libraries used include `yfinance` for data retrieval, `scikit-learn` for preprocessing, and `TensorFlow` for building and training the LSTM model.

---

## Libraries Used
- **`yfinance`**: For downloading historical stock data (adjusted close prices).  
- **`numpy`**: For numerical operations, such as array manipulations and calculations.  
- **`pandas`**: For handling time series data and data manipulation.  
- **`matplotlib`**: For plotting graphs and visualizations.  
- **`sklearn`**: For data scaling and evaluation metrics (e.g., RMSE).  
- **`tensorflow.keras`**: For building and training the LSTM model.  

---

## Key Functions
### Data Handling
1. **`download_data(tickers, start, end)`**  
   Downloads historical data for specified tickers from Yahoo Finance. Ensures data integrity by handling missing values.  

2. **`prepare_data(series, time_step)`**  
   Prepares the data for LSTM by normalizing and creating sequences based on a time step (default: 60 days).

### Modeling
3. **`build_lstm(input_shape)`**  
   Constructs an LSTM model with two hidden layers and dropout regularization to prevent overfitting.  

4. **`predict_lstm(model, data, scaler)`**  
   Predicts future prices using the trained LSTM model and scales them back to the original range.  

### Risk Assessment
5. **`historical_var(returns, confidence)`**  
   Calculates historical Value at Risk (VaR) at a specified confidence level.  

6. **`calculate_rmse(actual, predicted)`**  
   Computes the Root Mean Squared Error (RMSE) to evaluate model accuracy.  
---

## Workflow
1. **Data Download**: Retrieves adjusted close prices for ETF.  
2. **Portfolio Calculation**: Constructs a weighted portfolio.  
3. **Data Preprocessing**: Calculates portfolio returns, normalizes the data, and splits it into training and testing sets.  
4. **Model Training**: Trains the LSTM model using early stopping to avoid overfitting.  
5. **Evaluation**: Computes RMSE for training/testing and calculates historical and LSTM-based VaR.  
6. **Visualization**: Generates graphs for historical prices, predicted vs actual values, and future trends.  

---

## Results Example

#### Dataset Summary
- **Train set**: 2576 samples
- **Test set**: 644 samples

#### Model Performance
- **RMSE Train**: 1.7707  
- **RMSE Test**: 1.8595  
- **MAE Test**: 1.4751  

#### Value at Risk (VaR)
- **Historical VaR**: -0.0144  
- **LSTM VaR**: -0.0153  

#### Comparison of VaR and Expected Shortfall
| Method       | VaR (95%)   | Expected Shortfall (95%) |
|--------------|-------------|--------------------------|
| Historical   | -0.014380   | -0.020155               |
| Parametric   | -0.014997   | -0.017925               |
| Monte Carlo  | -0.014992   | -0.018981               |

---

## Visualization Examples

- **Historical Prices**: Plots showing the adjusted close prices.  
- **Prediction Performance**: Line graphs comparing actual vs predicted values.  
- **Future Price Trends**: Visualized forecasts for the portfolio over 6 months.  

---

## Disclaimer
This repository is provided for educational and informational purposes only. The content, code, and methodologies presented herein are not intended as financial or investment advice. 
The authors do not guarantee the accuracy, reliability, or completeness of the information provided.  
Use this repository at your own risk, and consult a qualified financial professional for investment decisions.  
Past performance is not indicative of future results, and machine learning models involve inherent uncertainties.  

By using this repository, you agree that the authors are not liable for any losses or damages resulting from the use of the content or tools provided.  

---
