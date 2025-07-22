---
layout: default
title: Transformer Model
---

# Transformer Model for Stock Price Prediction

This document provides an overview of the Transformer model implemented in this project for stock price prediction.

## Overview

The Transformer model is a neural network architecture that relies on attention mechanisms rather than recurrence. It was originally introduced for natural language processing tasks but has been adapted here for time series forecasting of stock prices.

## Key Features

- **Self-Attention Mechanism**: Captures relationships between data points regardless of their distance
- **Parallelization**: More efficient training compared to recurrent models
- **Positional Encoding**: Maintains sequence order information
- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence
- **Dimension Mismatch Handling**:
  - Automatically detects when input features don't match training features
  - Trims excess features when necessary
  - Retrains the scaler when receiving fewer features than expected
  - Provides detailed logging of dimension adjustments

## Architecture

The Transformer model implementation follows this structure:

1. **Data Preparation**:
   - Historical stock data fetching with Yahoo Finance
   - Technical indicator calculation
   - Feature engineering with temporal context
   - Sequence preparation

2. **Model Architecture**:
   ```
   - Input Layer
   - Positional Encoding
   - Transformer Encoder Layers (with self-attention)
   - Dense Layers
   - Output Layer
   ```

3. **Training Process**:
   - Adam optimizer with warmup and learning rate scheduling
   - Mean Squared Error loss function
   - Early stopping to prevent overfitting
   - Validation on hold-out data

4. **Prediction Pipeline**:
   - Load trained model and preprocessors
   - Process new data with the same feature engineering
   - Generate predictions for future time points

## Performance Evaluation

The Transformer model is evaluated using several metrics:

- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Measures average magnitude of errors
- **MAPE (Mean Absolute Percentage Error)**: Measures percentage error
- **R² (R-squared)**: Measures how well the model explains the target variance

## Usage Examples

```python
# Initialize Transformer predictor
from algorithms.Transformer import StockPriceTransformer

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

## Visualization

The Transformer implementation provides several visualizations:

1. **Training History**: Loss curves showing training and validation loss
2. **Prediction Accuracy**: Actual vs. predicted prices on test data
3. **Attention Maps**: Visualizations of what the model focuses on
4. **Trading Simulation**: Performance of a strategy based on predictions

## Comparison with Other Models

The Transformer model can be compared with other models in the project:
- **XGBoost**: Generally faster to train, better for tabular data with engineered features
- **LSTM**: Better suited for very long sequences and situations where memory of distant events is critical

Each model has different strengths, and their performance can vary depending on the specific stock and market conditions.

## Future Improvements

Potential improvements for the Transformer model:
- Incorporating market sentiment data
- Adding more sophisticated positional encodings
- Exploring hybrid models combining Transformer with other approaches
- Implementing a more advanced attention mechanism
