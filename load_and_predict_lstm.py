#!/usr/bin/env python3
"""
Load and predict using the LSTM model for stock price prediction.
This script demonstrates how to load a trained LSTM model and use it for prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms.LSTM import LSTMStockPredictor

def main():
    print("Stock Price Prediction - LSTM Model Loading Demo")
    print("=" * 50)
    
    # Initialize the LSTM predictor
    symbol = input("Enter stock symbol (default: AAPL): ") or "AAPL"
    
    predictor = LSTMStockPredictor(
        symbol=symbol,
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    # Check if model exists
    model_path = 'models/lstm_stock_model.h5'
    pipeline_path = f'models/{symbol}_lstm_pipeline.joblib'
    
    if os.path.exists(model_path) and os.path.exists(pipeline_path):
        print(f"\nLoading existing model for {symbol}...")
        predictor.load_model(
            model_path=model_path,
            load_pipeline=True,
            pipeline_path=pipeline_path
        )
    else:
        print(f"\nNo existing model found for {symbol}. Training a new model...")
        predictor.run_pipeline(
            window_size=60,
            epochs=20,  # Reduced for demo
            batch_size=32,
            patience=5,
            use_cached_data=True,
            cache_dir='data'
        )
    
    # Make predictions for the next few days
    days = int(input("\nHow many days to predict? (default: 5): ") or "5")
    
    future_predictions = predictor.predict_future(days=days)
    
    print(f"\nPredictions for {symbol} for the next {days} trading days:")
    print(future_predictions)
    
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_predictions.index, future_predictions['Predicted_Close'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    
    for i, row in future_predictions.iterrows():
        plt.annotate(f'${row["Predicted_Close"]:.2f}', 
                    (i, row["Predicted_Close"]),
                    xytext=(5, 10),
                    textcoords='offset points')
    
    plt.title(f'{symbol} - LSTM Price Predictions', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Predicted Stock Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(f'{symbol}_lstm_predictions.png')
    print(f"\nPrediction chart saved as {symbol}_lstm_predictions.png")
    plt.show()

if __name__ == "__main__":
    main()
