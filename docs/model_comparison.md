---
layout: default
title: Model Comparison
---

# Model Comparison

This page provides a detailed comparison of the three stock price prediction models implemented in this project: XGBoost, Transformer, and LSTM.

## Performance Metrics

The following metrics are based on testing with AAPL stock data over a 5-year period with a 20% test split:

| Model | Test RMSE | Test MAE | Test MAPE | Test R² |
|-------|-----------|----------|-----------|---------|
| XGBoost | 12.27 | 9.82 | 4.37% | 0.48 |
| Transformer | 15.02 | 12.92 | 9.76% | 0.37 |
| LSTM | 9.79 | 7.15 | 3.32% | 0.69 |

### Interpretation

- **RMSE (Root Mean Squared Error)**: Lower is better. LSTM shows the best performance.
- **MAE (Mean Absolute Error)**: Lower is better. LSTM shows the best performance.
- **MAPE (Mean Absolute Percentage Error)**: Lower is better. LSTM has the lowest percentage error.
- **R²**: Higher is better. LSTM explains the highest proportion of variance in the data.

## Strengths and Weaknesses

### XGBoost

**Strengths:**
- Faster training time
- Handles a mix of feature types well
- Good interpretability through feature importance
- Works well with engineered features
- Requires less data for good performance
- Robust dimension mismatch handling for feature count differences

**Weaknesses:**
- Limited ability to capture temporal dependencies
- May require more feature engineering
- Cannot inherently capture non-linear temporal patterns

### Transformer

**Strengths:**
- Captures complex patterns through attention mechanisms
- Parallel processing enables faster training compared to LSTM
- Effective for capturing long-range dependencies
- Less prone to vanishing gradient issues
- Sophisticated dimension mismatch handling with feature trimming and scaler retraining

**Weaknesses:**
- Requires more data to train effectively
- More complex to implement and fine-tune
- Computationally intensive
- Shows worse empirical performance in our tests

### LSTM

**Strengths:**
- Best empirical performance across all metrics
- Excellent for sequence modeling and time series
- Effectively captures temporal patterns
- Maintains memory of past events
- Feature dimension mismatch detection and handling

**Weaknesses:**
- Slower training due to sequential nature
- More prone to overfitting with small datasets
- Harder to interpret compared to XGBoost
- May suffer from vanishing gradient over very long sequences

## Training Time Comparison

| Model | Average Training Time |
|-------|----------------------|
| XGBoost | 2-5 minutes |
| Transformer | 10-20 minutes |
| LSTM | 5-15 minutes |

## Memory Usage

| Model | Approximate Model Size |
|-------|----------------------|
| XGBoost | 10-50 MB |
| Transformer | 50-200 MB |
| LSTM | 20-100 MB |

## Prediction Speed

| Model | Prediction Time for 5 Days |
|-------|---------------------------|
| XGBoost | < 1 second |
| Transformer | 1-2 seconds |
| LSTM | 1-2 seconds |

## Dimension Mismatch Handling

All models now include robust feature count mismatch detection and handling:

| Model | Excess Features Handling | Insufficient Features Handling |
|-------|--------------------------|--------------------------------|
| XGBoost | Trims extra features | Uses unscaled features when necessary |
| Transformer | Trims extra features | Retrains scaler on current feature set |
| LSTM | Trims extra features | Uses feature padding when possible |

This robust feature handling ensures models can make predictions even when the feature set changes between training and prediction time, which is common when using different stock symbols or when market conditions introduce new features.

## Use Case Recommendations

- **XGBoost**: Best for situations where interpretation is important, or when you have limited computational resources. Good for exploring feature importance and understanding market drivers.

- **Transformer**: Suitable for capturing complex patterns when you have substantial data and computational resources. May work better with certain stocks that have long-range dependencies.

- **LSTM**: Best overall performer for stock price prediction. Recommended for production use when prediction accuracy is the primary goal.

## Ensemble Approach

For maximum prediction accuracy, consider an ensemble approach that combines predictions from all three models, potentially weighted by their historical performance.

## Future Work

- Implement a voting ensemble of all three models
- Explore hybrid architectures that combine the strengths of each approach
- Add market sentiment analysis to complement technical indicators
- Incorporate transfer learning from models trained on related stocks
