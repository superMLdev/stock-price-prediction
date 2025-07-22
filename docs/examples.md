---
layout: default
title: Prediction Examples
---

# Stock Price Prediction Examples

This page provides practical examples of how to use the stock price prediction models for generating forecasts and making trading decisions.

## Quick Start Examples

Download these ready-to-use example scripts to get started quickly:

- [Complete Prediction Script]({{ site.baseurl }}/assets/examples/predict_stock_prices.py) - Compare predictions from all models
- [Autotrading Example]({{ site.baseurl }}/assets/examples/autotrading_example.py) - Simulate trading based on predictions

### Using the Example Scripts

The example scripts can be run from the command line:

```bash
# Make predictions using all models
python predict_stock_prices.py --symbol AAPL --days 5 --plot

# Run a trading simulation
python autotrading_example.py --symbol MSFT --model xgboost --initial_capital 10000 --plot
```

Both scripts accept various command-line arguments to customize their behavior:

**predict_stock_prices.py options:**
- `--symbol`: Stock ticker symbol (default: AAPL)
- `--days`: Number of days to predict (default: 5)
- `--period`: Historical data period (default: 1y)
- `--models`: Models to use (all, xgboost, lstm, transformer, or comma-separated list)
- `--plot`: Show prediction plots
- `--save`: Save plots to file

**autotrading_example.py options:**
- `--symbol`: Stock ticker symbol (default: AAPL)
- `--model`: Model to use (xgboost, lstm, or transformer)
- `--period`: Historical data period (default: 1y)
- `--initial_capital`: Starting capital for simulation (default: $10,000)
- `--threshold`: Price change threshold for signals (default: 0.01 = 1%)
- `--plot`: Show simulation plots
- `--save`: Save plots to file

## Basic Prediction Example

This basic example shows how to load a pre-trained model and make predictions for a stock:

```python
from algorithms.XGBoost import StockPricePredictor

# Initialize predictor with a stock symbol
predictor = StockPricePredictor(symbol="AAPL")

# Load a pre-trained model
predictor.load_model()

# Predict the next 5 days
future_predictions = predictor.predict_future(days=5)
print(future_predictions)

# Visualize predictions
predictor.plot_predictions()
```

## Model-Specific Examples

### XGBoost Example

```python
from algorithms.XGBoost import StockPricePredictor

# Initialize with your stock of interest
xgb_predictor = StockPricePredictor(
    symbol="MSFT",
    period="2y",
    test_size=0.2,
    use_cache=True
)

# Train a new model with feature engineering
xgb_predictor.train_model(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    use_technical_indicators=True
)

# Evaluate the model
metrics = xgb_predictor.evaluate_model()
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")

# Make and visualize predictions
predictions = xgb_predictor.predict_future(days=10)
xgb_predictor.plot_predictions(show_confidence=True)

# Save model for future use
xgb_predictor.save_model("msft_xgb_model.pkl")
```

### Transformer Example

```python
from algorithms.Transformer import StockPriceTransformer

# Initialize the transformer model
transformer = StockPriceTransformer(
    symbol="GOOGL",
    period="3y",
    window_size=30,
    test_size=0.15
)

# Train the model
transformer.train_model(
    epochs=50, 
    batch_size=32,
    learning_rate=0.001,
    num_heads=4,
    d_model=128
)

# Evaluate the model
transformer.evaluate_model()

# Make future predictions
transformer_predictions = transformer.predict_future(days=7)
transformer.plot_predictions()

# Save the model
transformer.save_model("googl_transformer")
```

### LSTM Example

```python
from algorithms.LSTM import LSTMStockPredictor

# Initialize LSTM predictor
lstm_predictor = LSTMStockPredictor(
    symbol="AMZN",
    period="1y",
    test_size=0.2
)

# Run the full pipeline (preprocessing, training, and evaluation)
lstm_predictor.run_pipeline(
    epochs=100,
    batch_size=64,
    lstm_units=50,
    dropout_rate=0.2
)

# Make future predictions
lstm_predictions = lstm_predictor.predict_future(days=5)
lstm_predictor.plot_predictions()
```

## Advanced Usage: Trading Signals

This example shows how to generate trading signals from predictions:

```python
from algorithms.XGBoost import StockPricePredictor
import pandas as pd
import matplotlib.pyplot as plt

# Load a pre-trained model
predictor = StockPricePredictor(symbol="AAPL")
predictor.load_model()

# Get historical data and predictions
historical_data = predictor.get_data()
predictions = predictor.predict()

# Combine actual and predicted data
trading_df = pd.DataFrame({
    'Actual': historical_data['Close'][-len(predictions):].values,
    'Predicted': predictions
})

# Generate simple trading signals
trading_df['Signal'] = 0  # 0 = hold
trading_df.loc[trading_df['Predicted'] > trading_df['Actual'] * 1.01, 'Signal'] = 1  # 1 = buy
trading_df.loc[trading_df['Predicted'] < trading_df['Actual'] * 0.99, 'Signal'] = -1  # -1 = sell

# Calculate potential returns (simplified)
trading_df['Return'] = 0.0
for i in range(1, len(trading_df)):
    if trading_df['Signal'].iloc[i-1] == 1:  # If previous signal was buy
        trading_df['Return'].iloc[i] = (trading_df['Actual'].iloc[i] / 
                                        trading_df['Actual'].iloc[i-1] - 1) * 100
    elif trading_df['Signal'].iloc[i-1] == -1:  # If previous signal was sell
        trading_df['Return'].iloc[i] = (trading_df['Actual'].iloc[i-1] / 
                                        trading_df['Actual'].iloc[i] - 1) * 100

# Plot signals and cumulative returns
plt.figure(figsize=(12, 8))

# Plot 1: Price and Predictions
plt.subplot(2, 1, 1)
plt.plot(trading_df.index, trading_df['Actual'], label='Actual Price', color='blue')
plt.plot(trading_df.index, trading_df['Predicted'], label='Predicted Price', color='green', linestyle='--')
buy_signals = trading_df[trading_df['Signal'] == 1]
sell_signals = trading_df[trading_df['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Actual'], color='green', marker='^', s=100, label='Buy')
plt.scatter(sell_signals.index, sell_signals['Actual'], color='red', marker='v', s=100, label='Sell')
plt.title('AAPL Price with Trading Signals')
plt.legend()

# Plot 2: Cumulative Returns
plt.subplot(2, 1, 2)
plt.plot(trading_df.index, trading_df['Return'].cumsum(), color='purple')
plt.title('Cumulative Returns (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print strategy statistics
total_trades = (trading_df['Signal'] != 0).sum()
profitable_trades = (trading_df['Return'] > 0).sum()
win_rate = profitable_trades / total_trades if total_trades > 0 else 0
cumulative_return = trading_df['Return'].sum()

print(f"Total Trades: {total_trades}")
print(f"Profitable Trades: {profitable_trades}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Cumulative Return: {cumulative_return:.2f}%")
```

## Combining Multiple Models

This example shows how to create an ensemble of models:

```python
from algorithms.XGBoost import StockPricePredictor
from algorithms.LSTM import LSTMStockPredictor
from algorithms.Transformer import StockPriceTransformer
import numpy as np
import matplotlib.pyplot as plt

# Initialize predictors for the same stock
symbol = "TSLA"
period = "1y"
test_size = 0.2

# Load or train individual models
xgb_predictor = StockPricePredictor(symbol=symbol, period=period, test_size=test_size)
xgb_predictor.load_model()  # or train_model()

lstm_predictor = LSTMStockPredictor(symbol=symbol, period=period, test_size=test_size)
lstm_predictor.load_model()  # or run_pipeline()

transformer_predictor = StockPriceTransformer(symbol=symbol, period=period, test_size=test_size)
transformer_predictor.load_model()  # or train_model()

# Make predictions with each model
days_to_predict = 5
xgb_preds = xgb_predictor.predict_future(days=days_to_predict)
lstm_preds = lstm_predictor.predict_future(days=days_to_predict)
transformer_preds = transformer_predictor.predict_future(days=days_to_predict)

# Create ensemble prediction (simple average)
ensemble_preds = (xgb_preds + lstm_preds + transformer_preds) / 3

# Display predictions
dates = xgb_predictor.get_future_dates(days=days_to_predict)
for i, date in enumerate(dates):
    print(f"Date: {date.strftime('%Y-%m-%d')}")
    print(f"  XGBoost: ${xgb_preds[i]:.2f}")
    print(f"  LSTM: ${lstm_preds[i]:.2f}")
    print(f"  Transformer: ${transformer_preds[i]:.2f}")
    print(f"  ENSEMBLE: ${ensemble_preds[i]:.2f}")
    print("-" * 40)

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(dates, xgb_preds, 'b-o', label='XGBoost')
plt.plot(dates, lstm_preds, 'g-o', label='LSTM')
plt.plot(dates, transformer_preds, 'r-o', label='Transformer')
plt.plot(dates, ensemble_preds, 'k-*', linewidth=2, markersize=10, label='Ensemble')
plt.title(f'{symbol} Price Predictions')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

These examples demonstrate the flexibility and power of the stock price prediction models included in this project. For more detailed information on each model, refer to their respective documentation pages.
