---
layout: default
title: API Reference
---

# API Reference

This page provides detailed documentation for the key classes and methods in the Stock Price Prediction project.

## XGBoost Model

### `StockPricePredictor` Class

```python
from algorithms.XGBoost import StockPricePredictor

predictor = StockPricePredictor(
    symbol="AAPL",
    period="5y",
    test_size=0.2,
    target_col='Close',
    prediction_horizon=1
)
```

#### Parameters

- `symbol` (str): Stock ticker symbol
- `period` (str): Data period to fetch (e.g., '5y' for 5 years)
- `test_size` (float): Proportion of data to use for testing (0-1)
- `target_col` (str): Column to predict (usually 'Close')
- `prediction_horizon` (int): Number of days ahead to predict

#### Key Methods

- `fetch_data(use_cached=True, cache_dir='data')`: Fetch historical stock data
- `engineer_features(lookback_days=30)`: Create features for training
- `preprocess_data(scale_features=True, scaler_type='robust')`: Preprocess data for model training
- `tune_hyperparameters(n_trials=100)`: Tune model hyperparameters with Optuna
- `train_model(tuned_params=None)`: Train XGBoost model
- `evaluate_model()`: Evaluate model performance
- `visualize_results(save_path=None)`: Create visualizations for model results
- `plot_trading_simulation()`: Create a trading simulation plot
- `backtest(test_periods=3)`: Perform walk-forward backtesting
- `save_model(filename="xgboost_stock_model.json", save_pipeline=True)`: Save the trained model
- `load_model(filename="xgboost_stock_model.json", load_pipeline=False)`: Load a trained model
- `predict_future(days=5, use_cached_data=True)`: Make predictions for future days
- `run_pipeline(tune=True, n_trials=50, visualize=True, ...)`: Run the complete pipeline

---

## LSTM Model

### `LSTMStockPredictor` Class

```python
from algorithms.LSTM import LSTMStockPredictor

predictor = LSTMStockPredictor(
    symbol="AAPL",
    period="5y",
    test_size=0.2,
    target_col='Close',
    prediction_horizon=1
)
```

#### Parameters

- `symbol` (str): Stock ticker symbol
- `period` (str): Data period to fetch (e.g., '5y' for 5 years)
- `test_size` (float): Proportion of data to use for testing (0-1)
- `target_col` (str): Column to predict (usually 'Close')
- `prediction_horizon` (int): Number of days ahead to predict

#### Key Methods

- `fetch_data(use_cached=True, cache_dir='data')`: Fetch historical stock data
- `add_technical_indicators(df)`: Add technical indicators to the dataframe
- `engineer_features(lookback_days=30)`: Create features for LSTM training
- `create_sequences(X, y, window_size=None)`: Create sequences for LSTM input
- `preprocess_data(window_size=60, scale_features=True)`: Preprocess data for LSTM
- `build_model(input_shape)`: Build LSTM model architecture
- `train_model(epochs=50, batch_size=32, patience=10)`: Train LSTM model
- `evaluate_model()`: Evaluate model performance
- `visualize_results(save_path=None)`: Create visualizations for model results
- `plot_trading_simulation()`: Create a trading simulation plot
- `save_model(model_path='models/lstm_stock_model.h5', ...)`: Save the trained model
- `load_model(model_path='models/lstm_stock_model.h5', ...)`: Load a trained model
- `predict_future(days=5, use_cached_data=True)`: Make predictions for future days
- `run_pipeline(window_size=60, epochs=50, batch_size=32, ...)`: Run the complete pipeline

---

## Transformer Model

### `StockPriceTransformer` Class

```python
from algorithms.Transformer import StockPriceTransformer

predictor = StockPriceTransformer(
    symbol="AAPL",
    period="5y"
)
```

#### Parameters

- `symbol` (str): Stock ticker symbol
- `period` (str): Data period to fetch (e.g., '5y' for 5 years)

#### Key Methods

- `fetch_data(use_cached=True, cache_dir='data')`: Fetch historical stock data
- `engineer_features()`: Create features for Transformer training
- `preprocess_data()`: Preprocess data for model training
- `build_model()`: Build Transformer model architecture
- `train_model()`: Train Transformer model
- `evaluate_model()`: Evaluate model performance
- `visualize_results()`: Create visualizations for model results
- `save_model()`: Save the trained model
- `load_model()`: Load a trained model
- `predict_future(days=5)`: Make predictions for future days
- `run_pipeline()`: Run the complete pipeline

---

## Utility Functions

### Comparison Script

The `compare_all_models.py` script provides functions for comparing the performance of all three models:

- `load_xgboost_model()`: Load the XGBoost model
- `load_transformer_model()`: Load the Transformer model
- `load_lstm_model()`: Load the LSTM model
- `train_lstm_model()`: Train a new LSTM model if one doesn't exist
- `compare_predictions()`: Compare predictions from all models
- `create_comparison_chart()`: Create visualizations comparing model performance
- `compare_prediction_charts()`: Compare prediction charts across models

### Helper Scripts

- `load_and_predict_lstm.py`: Interactive script for using LSTM models
