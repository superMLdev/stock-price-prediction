# LSTM Model for Stock Price Prediction

This document provides an overview of the Long Short-Term Memory (LSTM) model implemented in this project for stock price prediction.

## Overview

The LSTM model is a recurrent neural network architecture designed specifically for sequence modeling and time series prediction. Unlike traditional neural networks, LSTM networks have feedback connections and can process entire sequences of data, making them well-suited for stock price prediction.

## Key Features

- **Memory Cells**: LSTM maintains information over long sequences
- **Sequential Data Processing**: Processes time-ordered stock price data
- **Feature Engineering**: Technical indicators and time-lagged features
- **Scalability**: Can be trained on various stock symbols and timeframes
- **Trading Simulation**: Includes a simulated trading strategy based on predictions

## Architecture

The LSTM model implementation follows this structure:

1. **Data Preparation**:
   - Historical stock data fetching with Yahoo Finance
   - Technical indicator calculation (MACD, RSI, Bollinger Bands, etc.)
   - Feature engineering with lookback periods
   - Time series sequence creation

2. **Model Architecture**:
   ```
   Sequential([
     LSTM(100, return_sequences=True, input_shape=input_shape),
     Dropout(0.2),
     LSTM(50, return_sequences=False),
     Dropout(0.2),
     Dense(25, activation='relu'),
     Dense(1)
   ])
   ```

3. **Training Process**:
   - Adam optimizer with learning rate scheduling
   - Mean Squared Error loss function
   - Early stopping to prevent overfitting
   - Validation on hold-out data

4. **Prediction Pipeline**:
   - Load trained model and scalers
   - Process new data with the same feature engineering
   - Generate sequence-based predictions

## Performance Evaluation

The LSTM model is evaluated using several metrics:

- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Measures average magnitude of errors
- **MAPE (Mean Absolute Percentage Error)**: Measures percentage error
- **R² (R-squared)**: Measures how well the model explains the target variance

## Usage Examples

```python
# Initialize LSTM predictor
from algorithms.LSTM import LSTMStockPredictor

predictor = LSTMStockPredictor(
    symbol="AAPL",
    period="5y",
    test_size=0.2,
    target_col='Close',
    prediction_horizon=1
)

# Train model
model, metrics = predictor.run_pipeline(
    window_size=60,
    epochs=50,
    batch_size=32,
    patience=10
)

# Make future predictions
future_predictions = predictor.predict_future(days=5)
print(future_predictions)
```

## Visualization

The LSTM implementation provides several visualizations:

1. **Training History**: Loss curves showing training and validation loss
2. **Prediction Accuracy**: Actual vs. predicted prices on test data
3. **Residual Analysis**: Error distribution and patterns
4. **Trading Simulation**: Performance of a strategy based on predictions

## Model Persistence

The model is saved in two parts:
- The neural network architecture and weights (`.h5` format)
- The pipeline components including scalers and configuration (`.joblib` format)

This allows for easy loading and reuse of trained models.

## Trading Strategy

A simple trading strategy is implemented based on the LSTM predictions:
- Buy when the predicted price increases
- Sell when the predicted price decreases

The performance of this strategy is compared against a buy-and-hold baseline.

## Comparison with Other Models

The LSTM model can be compared with other models in the project:
- **XGBoost**: Generally faster to train, potentially more accurate for short-term predictions
- **Transformer**: Better at capturing longer-term dependencies and complex patterns

Each model has different strengths, and their performance can vary depending on the specific stock and market conditions.

## Future Improvements

Potential improvements for the LSTM model:
- Bidirectional LSTM layers for better context understanding
- Attention mechanisms to focus on important time periods
- More sophisticated feature engineering
- Ensemble methods combining LSTM with other models
