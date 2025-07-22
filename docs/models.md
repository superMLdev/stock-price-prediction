---
layout: default
title: Models
---

# Stock Price Prediction Models

This project implements three different machine learning models for stock price prediction, each with its own strengths and use cases.

## [XGBoost](xgboost.html)

XGBoost is a gradient boosting framework that uses decision trees as base learners. Our implementation includes:

- Extensive feature engineering with technical indicators
- Hyperparameter tuning with Optuna
- Robust evaluation and visualization
- Trading simulation capabilities

[Learn more about our XGBoost implementation](xgboost.html)

## [Transformer](transformer.html)

The Transformer model uses self-attention mechanisms to handle sequential data. Key features:

- Self-attention mechanism for capturing complex patterns
- Positional encoding to maintain sequence information
- Multi-head attention for focusing on different aspects of the input
- Scalable architecture for handling large datasets

[Learn more about our Transformer implementation](transformer.html)

## [LSTM](lstm.html)

Long Short-Term Memory (LSTM) networks are specifically designed for time series forecasting. Our implementation includes:

- Memory cells that retain information over long sequences
- Advanced sequence processing for time-ordered stock data
- Technical indicators and time-lagged features
- Trading simulation and evaluation

[Learn more about our LSTM implementation](lstm.html)

## [Model Comparison](model_comparison.html)

For a detailed comparison of all three models, including performance metrics, strengths, and weaknesses, visit our [Model Comparison](model_comparison.html) page.
