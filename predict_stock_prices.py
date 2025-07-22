#!/usr/bin/env python3
"""
Complete example script for stock price prediction using all three models.

This script demonstrates how to:
1. Load pre-trained models
2. Make predictions with each model
3. Create an ensemble prediction
4. Visualize and compare results

Usage:
    python predict_stock_prices.py --symbol AAPL --days 5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

try:
    from algorithms.XGBoost import StockPricePredictor
    from algorithms.LSTM import LSTMStockPredictor
    from algorithms.Transformer import StockPriceTransformer
except ImportError:
    print("Error: Could not import prediction models.")
    print("Make sure you're running this script from the project directory.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Price Prediction Example')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict (default: 5)')
    parser.add_argument('--period', type=str, default='1y', help='Historical data period (default: 1y)')
    parser.add_argument('--models', type=str, default='all', 
                        help='Models to use (all, xgboost, lstm, transformer, or comma-separated list)')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save', action='store_true', help='Save plots')
    return parser.parse_args()


def load_models(symbol, period, test_size=0.2, models='all'):
    """Load or train the specified models."""
    model_dict = {}
    models_to_load = models.lower().split(',') if models != 'all' else ['xgboost', 'lstm', 'transformer']
    
    try:
        if 'xgboost' in models_to_load:
            print(f"Loading XGBoost model for {symbol}...")
            xgb_predictor = StockPricePredictor(symbol=symbol, period=period, test_size=test_size)
            try:
                xgb_predictor.load_model()
            except:
                print("Pre-trained XGBoost model not found. Training new model...")
                xgb_predictor.train_model()
            model_dict['XGBoost'] = xgb_predictor
        
        if 'lstm' in models_to_load:
            print(f"Loading LSTM model for {symbol}...")
            lstm_predictor = LSTMStockPredictor(symbol=symbol, period=period, test_size=test_size)
            try:
                lstm_predictor.load_model()
            except:
                print("Pre-trained LSTM model not found. Training new model...")
                lstm_predictor.run_pipeline()
            model_dict['LSTM'] = lstm_predictor
        
        if 'transformer' in models_to_load:
            print(f"Loading Transformer model for {symbol}...")
            transformer_predictor = StockPriceTransformer(symbol=symbol, period=period, test_size=test_size)
            try:
                transformer_predictor.load_model()
            except:
                print("Pre-trained Transformer model not found. Training new model...")
                transformer_predictor.train_model()
            model_dict['Transformer'] = transformer_predictor
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        sys.exit(1)
    
    return model_dict


def make_predictions(model_dict, days):
    """Generate predictions using each model."""
    predictions = {}
    dates = None
    
    for model_name, model in model_dict.items():
        print(f"Making predictions with {model_name} model...")
        preds = model.predict_future(days=days)
        predictions[model_name] = preds
        
        # Use the dates from any model (they should be the same)
        if dates is None:
            dates = model.get_future_dates(days=days)
    
    # Create ensemble prediction if we have multiple models
    if len(model_dict) > 1:
        ensemble_preds = np.zeros(days)
        for model_preds in predictions.values():
            ensemble_preds += model_preds
        ensemble_preds /= len(model_dict)
        predictions['Ensemble'] = ensemble_preds
    
    return predictions, dates


def display_predictions(predictions, dates):
    """Display the predictions in a formatted table."""
    print("\n" + "="*50)
    print(f"Stock Price Predictions")
    print("="*50)
    
    # Header row
    header = "Date"
    for model in predictions.keys():
        header += f" | {model:>10}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for i, date in enumerate(dates):
        row = f"{date.strftime('%Y-%m-%d')}"
        for model, preds in predictions.items():
            row += f" | ${preds[i]:10.2f}"
        print(row)
    
    print("="*50 + "\n")


def plot_predictions(predictions, dates, symbol, save=False):
    """Visualize the predictions."""
    plt.figure(figsize=(12, 6))
    
    colors = {
        'XGBoost': 'blue',
        'LSTM': 'green',
        'Transformer': 'red',
        'Ensemble': 'black'
    }
    
    markers = {
        'XGBoost': 'o',
        'LSTM': 's',
        'Transformer': '^',
        'Ensemble': '*'
    }
    
    for model, preds in predictions.items():
        plt.plot(dates, preds, 
                 color=colors.get(model, 'gray'), 
                 marker=markers.get(model, 'o'),
                 linewidth=2 if model == 'Ensemble' else 1.5,
                 markersize=8 if model == 'Ensemble' else 6,
                 label=model)
    
    plt.title(f'{symbol} Stock Price Predictions', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save:
        filename = f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")
    
    plt.show()


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"\nStock Price Prediction Example")
    print(f"Symbol: {args.symbol}")
    print(f"Days to predict: {args.days}")
    print(f"Models: {args.models}")
    
    # Load models
    model_dict = load_models(args.symbol, args.period, models=args.models)
    
    if not model_dict:
        print("No models were loaded. Exiting.")
        return
    
    # Make predictions
    predictions, dates = make_predictions(model_dict, args.days)
    
    # Display predictions
    display_predictions(predictions, dates)
    
    # Plot if requested
    if args.plot:
        plot_predictions(predictions, dates, args.symbol, save=args.save)
    
    print("Prediction complete!")


if __name__ == "__main__":
    main()
