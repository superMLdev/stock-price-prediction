#!/usr/bin/env python3
"""
Automatic trading example using stock price predictions.

This script demonstrates how to:
1. Generate predictions using a trained model
2. Create trading signals based on the predictions
3. Simulate a trading strategy
4. Evaluate the strategy's performance

Usage:
    python autotrading_example.py --symbol AAPL --model xgboost --initial_capital 10000
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

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
    parser = argparse.ArgumentParser(description='Stock Trading Simulator')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--model', type=str, default='xgboost', 
                        choices=['xgboost', 'lstm', 'transformer'], 
                        help='Model to use for predictions')
    parser.add_argument('--period', type=str, default='1y', help='Historical data period (default: 1y)')
    parser.add_argument('--initial_capital', type=float, default=10000, 
                        help='Initial capital for trading simulation (default: $10,000)')
    parser.add_argument('--threshold', type=float, default=0.01, 
                        help='Price change threshold for generating signals (default: 0.01 = 1%)')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--save', action='store_true', help='Save plots')
    return parser.parse_args()


def load_model(symbol, period, model_type, test_size=0.2):
    """Load the specified prediction model."""
    try:
        if model_type == 'xgboost':
            print(f"Loading XGBoost model for {symbol}...")
            predictor = StockPricePredictor(symbol=symbol, period=period, test_size=test_size)
            try:
                predictor.load_model()
            except:
                print("Pre-trained model not found. Training new model...")
                predictor.train_model()
        
        elif model_type == 'lstm':
            print(f"Loading LSTM model for {symbol}...")
            predictor = LSTMStockPredictor(symbol=symbol, period=period, test_size=test_size)
            try:
                predictor.load_model()
            except:
                print("Pre-trained model not found. Training new model...")
                predictor.run_pipeline()
        
        elif model_type == 'transformer':
            print(f"Loading Transformer model for {symbol}...")
            predictor = StockPriceTransformer(symbol=symbol, period=period, test_size=test_size)
            try:
                predictor.load_model()
            except:
                print("Pre-trained model not found. Training new model...")
                predictor.train_model()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    return predictor


def generate_trading_signals(predictor, threshold=0.01):
    """Generate trading signals based on prediction-actual difference."""
    # Get historical data and predictions
    data = predictor.get_data()
    predictions = predictor.predict()
    
    # Extract test portion that matches predictions
    actual_prices = data['Close'][-len(predictions):].values
    
    # Create trading dataframe
    trading_df = pd.DataFrame({
        'Date': data.index[-len(predictions):],
        'Actual': actual_prices,
        'Predicted': predictions,
        'NextDayPredicted': np.append(predictions[1:], np.nan)
    })
    
    # Drop the last row as it doesn't have a next-day prediction
    trading_df = trading_df.iloc[:-1].copy()
    
    # Calculate predicted percent change
    trading_df['PredictedChange'] = (trading_df['NextDayPredicted'] / trading_df['Actual']) - 1
    
    # Generate signals based on predicted change
    trading_df['Signal'] = 0  # 0 = hold
    trading_df.loc[trading_df['PredictedChange'] > threshold, 'Signal'] = 1  # 1 = buy
    trading_df.loc[trading_df['PredictedChange'] < -threshold, 'Signal'] = -1  # -1 = sell
    
    return trading_df


def simulate_trading(trading_df, initial_capital=10000.0):
    """Simulate trading based on the signals."""
    # Copy the dataframe
    simulation = trading_df.copy()
    
    # Initialize portfolio metrics
    simulation['Capital'] = initial_capital
    simulation['Shares'] = 0
    simulation['PortfolioValue'] = initial_capital
    simulation['Return'] = 0.0
    
    # Calculate actual price changes
    simulation['ActualChange'] = simulation['Actual'].pct_change()
    simulation['ActualChange'].iloc[0] = 0
    
    position = 0  # 0 = cash, 1 = long, -1 = short
    
    # Simulate trading
    for i in range(len(simulation)):
        if i == 0:
            continue
        
        prev_capital = simulation['Capital'].iloc[i-1]
        prev_shares = simulation['Shares'].iloc[i-1]
        current_price = simulation['Actual'].iloc[i]
        
        # Execute previous day's signal
        signal = simulation['Signal'].iloc[i-1]
        
        if signal == 1 and position != 1:  # Buy signal
            # Close any short position
            if position == -1:
                short_profit = prev_shares * (simulation['Actual'].iloc[i-1] - current_price)
                prev_capital += short_profit
            
            # Go long
            shares_to_buy = prev_capital / current_price
            simulation['Shares'].iloc[i] = shares_to_buy
            simulation['Capital'].iloc[i] = 0
            position = 1
        
        elif signal == -1 and position != -1:  # Sell signal
            # Close any long position
            if position == 1:
                sale_value = prev_shares * current_price
                prev_capital = sale_value
            
            # Go short (simplified - just track notional short value)
            shares_to_short = prev_capital / current_price
            simulation['Shares'].iloc[i] = -shares_to_short  # Negative shares for short
            simulation['Capital'].iloc[i] = prev_capital * 2  # Double capital (original + shorted)
            position = -1
        
        else:  # Hold current position
            simulation['Shares'].iloc[i] = prev_shares
            simulation['Capital'].iloc[i] = prev_capital
        
        # Calculate portfolio value
        if position == 1:  # Long position
            simulation['PortfolioValue'].iloc[i] = simulation['Shares'].iloc[i] * current_price
        elif position == -1:  # Short position
            simulation['PortfolioValue'].iloc[i] = simulation['Capital'].iloc[i] - (simulation['Shares'].iloc[i] * current_price)
        else:  # Cash position
            simulation['PortfolioValue'].iloc[i] = simulation['Capital'].iloc[i]
        
        # Calculate daily return
        simulation['Return'].iloc[i] = (simulation['PortfolioValue'].iloc[i] / 
                                         simulation['PortfolioValue'].iloc[i-1] - 1) * 100
    
    # Calculate cumulative metrics
    simulation['CumulativeReturn'] = (1 + simulation['Return'] / 100).cumprod() - 1
    simulation['DrawdownPct'] = simulation['PortfolioValue'] / simulation['PortfolioValue'].cummax() - 1
    
    return simulation


def calculate_performance_metrics(simulation):
    """Calculate trading strategy performance metrics."""
    total_days = len(simulation)
    total_trades = (simulation['Signal'] != 0).sum()
    positive_returns = (simulation['Return'] > 0).sum()
    negative_returns = (simulation['Return'] < 0).sum()
    
    win_rate = positive_returns / total_trades if total_trades > 0 else 0
    total_return = simulation['CumulativeReturn'].iloc[-1] * 100
    annualized_return = ((1 + simulation['CumulativeReturn'].iloc[-1]) ** (365 / total_days) - 1) * 100
    
    daily_returns = simulation['Return']
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    max_drawdown = simulation['DrawdownPct'].min() * 100
    
    return {
        'Total Trades': total_trades,
        'Win Rate': f"{win_rate:.2%}",
        'Total Return': f"{total_return:.2f}%",
        'Annualized Return': f"{annualized_return:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2f}%",
    }


def plot_trading_simulation(simulation, symbol, model_type, save=False):
    """Plot trading simulation results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Price and Signals
    axes[0].plot(simulation['Date'], simulation['Actual'], label='Actual Price', color='blue')
    axes[0].plot(simulation['Date'], simulation['Predicted'], label='Predicted Price', color='green', alpha=0.7, linestyle='--')
    
    buy_signals = simulation[simulation['Signal'] == 1]
    sell_signals = simulation[simulation['Signal'] == -1]
    
    if not buy_signals.empty:
        axes[0].scatter(buy_signals['Date'], buy_signals['Actual'], color='green', marker='^', s=100, label='Buy Signal')
    
    if not sell_signals.empty:
        axes[0].scatter(sell_signals['Date'], sell_signals['Actual'], color='red', marker='v', s=100, label='Sell Signal')
    
    axes[0].set_title(f'{symbol} Trading Simulation using {model_type.upper()} Model', fontsize=16)
    axes[0].set_ylabel('Price ($)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Portfolio Value
    axes[1].plot(simulation['Date'], simulation['PortfolioValue'], color='purple', label='Portfolio Value')
    axes[1].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    # Plot 3: Drawdown
    axes[2].fill_between(simulation['Date'], simulation['DrawdownPct'] * 100, 0, color='red', alpha=0.3)
    axes[2].set_ylabel('Drawdown (%)', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f"{symbol}_{model_type}_trading_simulation_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename, dpi=300)
        print(f"Plot saved as {filename}")
    
    plt.show()


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"\nStock Trading Simulation")
    print(f"Symbol: {args.symbol}")
    print(f"Model: {args.model}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    
    # Load model
    predictor = load_model(args.symbol, args.period, args.model)
    
    # Generate trading signals
    trading_df = generate_trading_signals(predictor, threshold=args.threshold)
    
    # Simulate trading
    simulation = simulate_trading(trading_df, initial_capital=args.initial_capital)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(simulation)
    
    # Display results
    print("\n" + "="*50)
    print(f"Trading Strategy Performance")
    print("="*50)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    print("="*50)
    
    # Plot results
    if args.plot:
        plot_trading_simulation(simulation, args.symbol, args.model, save=args.save)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
