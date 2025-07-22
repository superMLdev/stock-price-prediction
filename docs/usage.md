---
layout: default
title: Usage Guide
---

# Usage Guide

This guide provides instructions on how to use the Stock Price Prediction project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/superMLdev/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Data

The project automatically fetches stock data using the Yahoo Finance API. You can customize the stock symbol, time period, and other parameters when initializing the models.

Data will be cached in the `data/` directory to avoid redundant downloads.

## Basic Usage

### Comparing All Models

To compare all three models (XGBoost, Transformer, and LSTM):

```bash
python compare_all_models.py
```

This script will:
1. Load or train each model
2. Generate predictions for the next 5 trading days
3. Compare model performance metrics
4. Create visualization charts of the results

### Working with Individual Models

#### XGBoost

```python
from algorithms.XGBoost import StockPricePredictor

# Initialize the predictor
predictor = StockPricePredictor(
    symbol="AAPL",             # Stock ticker symbol
    period="5y",               # Data period to fetch
    test_size=0.2,             # Proportion of data for testing
    target_col='Close',        # Target column to predict
    prediction_horizon=1       # Days ahead to predict
)

# Run the complete pipeline
model, metrics, importance = predictor.run_pipeline(
    tune=True,                 # Use Optuna for hyperparameter tuning
    n_trials=50,               # Number of Optuna trials
    visualize=True,            # Generate visualizations
    save_model=True,           # Save the model
    backtest=True,             # Perform backtesting
    scale_features=True,       # Scale the features
    use_cached_data=True,      # Use cached data if available
    cache_dir='data'           # Directory to store/retrieve cached data
)

# Make predictions for the next 5 days
future_predictions = predictor.predict_future(days=5)
print(future_predictions)
```

#### LSTM

```python
from algorithms.LSTM import LSTMStockPredictor

# Initialize the predictor
predictor = LSTMStockPredictor(
    symbol="AAPL",
    period="5y",
    test_size=0.2,
    target_col='Close',
    prediction_horizon=1
)

# Run the pipeline
model, metrics = predictor.run_pipeline(
    window_size=60,
    epochs=50,
    batch_size=32,
    patience=10,
    use_cached_data=True,
    cache_dir='data'
)

# Make future predictions
future_predictions = predictor.predict_future(days=5)
print(future_predictions)
```

#### Transformer

```python
from algorithms.Transformer import StockPriceTransformer

# Initialize the predictor
predictor = StockPriceTransformer(
    symbol="AAPL",
    period="5y"
)

# Train model
predictor.run_pipeline()

# Make future predictions
future_predictions = predictor.predict_future(days=5)
print(future_predictions)
```

## Loading Pre-trained Models

### Loading an XGBoost Model

```python
from algorithms.XGBoost import StockPricePredictor

# Initialize the predictor
predictor = StockPricePredictor(symbol="AAPL")

# Load the saved model and pipeline
predictor.load_model(
    load_pipeline=True,
    model_dir='models',
    filename='AAPL_pipeline.joblib'
)

# Make predictions with the loaded model
future_predictions = predictor.predict_future(days=5)
print(future_predictions)
```

### Loading an LSTM Model

```python
from algorithms.LSTM import LSTMStockPredictor

# Initialize the predictor
predictor = LSTMStockPredictor(symbol="AAPL")

# Load the saved model and pipeline
predictor.load_model(
    model_path='models/lstm_stock_model.h5',
    load_pipeline=True,
    pipeline_path='models/AAPL_lstm_pipeline.joblib'
)

# Make predictions with the loaded model
future_predictions = predictor.predict_future(days=5)
print(future_predictions)
```

## Helper Scripts

### Load and Predict with LSTM

The project includes a helper script for using LSTM models:

```bash
python load_and_predict_lstm.py
```

This interactive script allows you to:
1. Enter a stock symbol
2. Load an existing model or train a new one if not available
3. Specify the number of days to predict
4. View the predictions and visualizations
