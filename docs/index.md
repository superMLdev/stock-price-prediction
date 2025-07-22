---
layout: default
title: Stock Price Prediction
---

# Stock Price Prediction

This project implements multiple machine learning models for stock price prediction, including XGBoost, Transformer, and LSTM approaches.

## Overview

The Stock Price Prediction project offers a comprehensive framework for forecasting stock prices using various machine learning techniques. Each model is implemented with a complete pipeline from data acquisition to model evaluation and visualization.

## Key Features

- **Multiple Models**: XGBoost, Transformer, and LSTM implementations
- **Data Pipeline**: Automated data fetching from Yahoo Finance
- **Feature Engineering**: Technical indicators and custom features
- **Hyperparameter Tuning**: Optimization for model parameters
- **Visualization**: Performance metrics and trading simulations
- **Model Persistence**: Save and load trained models

## Models

### [XGBoost](xgboost.html)

A gradient boosting framework that uses decision trees as base learners. The XGBoost model is optimized for stock price prediction with extensive feature engineering.

### [Transformer](transformer.html)

A neural network architecture based on the transformer model, which uses self-attention mechanisms to handle sequential data.

### [LSTM](lstm.html)

Long Short-Term Memory network specifically designed for time series forecasting with the ability to capture long-term dependencies.

## Getting Started

To get started with this project, check out the:

- [Usage Guide](usage.html): Installation and basic usage
- [Examples](examples.html): Practical examples and downloadable scripts
- [Model Documentation](models.html): Detailed explanation of each model

## Performance Comparison

The models have been evaluated on various stocks, with performance metrics including RMSE, MAE, MAPE, and R². For a detailed comparison, see the [model comparison](model_comparison.html) page.
