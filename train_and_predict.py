#!/usr/bin/env python3
"""
train_and_predict.py - CLI tool for stock price prediction

This script provides a command-line interface for:
1. Training stock price prediction models (XGBoost, LSTM, Transformer)
2. Making future price predictions with trained models
3. Visualizing prediction results

Usage:
    python train_and_predict.py --symbol AAPL --models xgboost,lstm --days 5 --train
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_and_predict.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from algorithms.XGBoost import StockPricePredictor
    from algorithms.LSTM import LSTMStockPredictor
    from algorithms.Transformer import StockPriceTransformer
except ImportError:
    logger.error("Could not import prediction models.")
    logger.error("Make sure you're running this script from the project directory.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train stock prediction models and make price predictions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock symbol (e.g., AAPL, MSFT, GOOG)')
    
    # Model selection
    parser.add_argument('--models', type=str, default='xgboost',
                        help='Models to use: xgboost, lstm, transformer, or all (comma-separated)')
    
    # Training options
    parser.add_argument('--train', action='store_true',
                        help='Force training new models even if saved models exist')
    parser.add_argument('--period', type=str, default='5y',
                        help='Data period for training (e.g., 1y, 2y, 5y, max)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (0-1)')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning (for XGBoost)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of hyperparameter tuning trials')
    
    # Prediction options
    parser.add_argument('--days', type=int, default=5,
                        help='Number of days to predict')
    
    # Visualization options
    parser.add_argument('--plot', action='store_true',
                        help='Show prediction plots')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save prediction plots')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save predictions to CSV file')
    
    # Performance evaluation
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtesting on historical data')
    
    return parser.parse_args()


def train_model(model_type, symbol, period, test_size, force_train=False, tune=False, n_trials=50):
    """
    Train or load a stock prediction model.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('xgboost', 'lstm', or 'transformer')
    symbol : str
        Stock symbol
    period : str
        Data period for training
    test_size : float
        Proportion of data to use for testing
    force_train : bool
        Whether to force training even if saved model exists
    tune : bool
        Whether to perform hyperparameter tuning
    n_trials : int
        Number of trials for hyperparameter tuning
    
    Returns:
    --------
    object
        Trained model instance
    """
    logger.info(f"Preparing {model_type.upper()} model for {symbol}...")
    
    try:
        if model_type.lower() == 'xgboost':
            model = StockPricePredictor(
                symbol=symbol,
                period=period,
                test_size=test_size
            )
            
            # First, try to fetch data to verify it exists
            try:
                data = model.fetch_data(use_cached=True)
                if data is None or data.empty:
                    logger.error(f"No data available for {symbol}. Please check the symbol and try again.")
                    return None
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None
                
            if force_train:
                logger.info("Training new XGBoost model...")
                if tune:
                    logger.info(f"Performing hyperparameter tuning with {n_trials} trials...")
                    best_params = model.tune_hyperparameters(n_trials=n_trials)
                    model.train_model(tuned_params=best_params)
                else:
                    model.train_model()
                model.evaluate_model()
                model.save_model(save_pipeline=True)
            else:
                try:
                    logger.info("Attempting to load saved XGBoost model...")
                    model.load_model(load_pipeline=True)
                    logger.info("Loaded existing XGBoost model")
                except Exception as e:
                    logger.warning(f"Could not load saved model: {e}")
                    logger.info("Training new XGBoost model...")
                    if tune:
                        best_params = model.tune_hyperparameters(n_trials=n_trials)
                        model.train_model(tuned_params=best_params)
                    else:
                        model.train_model()
                    model.evaluate_model()
                    model.save_model(save_pipeline=True)
            
        elif model_type.lower() == 'lstm':
            model = LSTMStockPredictor(
                symbol=symbol,
                period=period,
                test_size=test_size
            )
            
            # First, try to fetch data to verify it exists
            try:
                data = model.fetch_data(use_cached=True)
                if data is None or data.empty:
                    logger.error(f"No data available for {symbol}. Please check the symbol and try again.")
                    return None
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None
                
            if force_train:
                logger.info("Training new LSTM model...")
                model.run_pipeline()
            else:
                try:
                    logger.info("Attempting to load saved LSTM model...")
                    model.load_model()
                    logger.info("Loaded existing LSTM model")
                except Exception as e:
                    logger.warning(f"Could not load saved model: {e}")
                    logger.info("Training new LSTM model...")
                    model.run_pipeline()
            
        elif model_type.lower() == 'transformer':
            model = StockPriceTransformer(
                symbol=symbol,
                period=period,
                test_size=test_size
            )
            
            # First, try to fetch data to verify it exists
            try:
                data = model.fetch_data(use_cached=True)
                if data is None or data.empty:
                    logger.error(f"No data available for {symbol}. Please check the symbol and try again.")
                    return None
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None
                
            if force_train:
                logger.info("Training new Transformer model...")
                model.train_model()
            else:
                try:
                    logger.info("Attempting to load saved Transformer model...")
                    model.load_model()
                    logger.info("Loaded existing Transformer model")
                except Exception as e:
                    logger.warning(f"Could not load saved model: {e}")
                    logger.info("Training new Transformer model...")
                    model.train_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training/loading {model_type} model: {str(e)}")
        return None


def make_predictions(models_dict, days):
    """
    Generate predictions using each model.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary mapping model names to model instances
    days : int
        Number of days to predict
    
    Returns:
    --------
    tuple
        (predictions, dates)
    """
    predictions = {}
    dates = None
    
    for model_name, model in models_dict.items():
        try:
            logger.info(f"Making predictions with {model_name} model...")
            start_time = time.time()
            preds = model.predict_future(days=days)
            
            # If model returns a list, keep it; if it returns a DataFrame, extract values
            if isinstance(preds, pd.DataFrame):
                preds = preds.values.flatten()
                
            predictions[model_name] = preds
            
            # Use the dates from any model (they should be the same)
            if dates is None:
                dates = model.get_future_dates(days=days)
                
            logger.info(f"Prediction with {model_name} completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {str(e)}")
    
    # Create ensemble prediction if we have multiple models
    if len(models_dict) > 1:
        try:
            ensemble_preds = np.zeros(days)
            for model_name, model_preds in predictions.items():
                ensemble_preds += model_preds
            ensemble_preds /= len(models_dict)
            predictions['Ensemble'] = ensemble_preds
            logger.info("Created ensemble prediction")
        except Exception as e:
            logger.error(f"Error creating ensemble prediction: {str(e)}")
    
    return predictions, dates


def display_predictions(predictions, dates, symbol):
    """
    Display the predictions in a formatted table.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary mapping model names to prediction arrays
    dates : list
        List of dates for predictions
    symbol : str
        Stock symbol
    """
    print("\n" + "="*50)
    print(f"{symbol} Stock Price Predictions")
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
    """
    Visualize the predictions.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary mapping model names to prediction arrays
    dates : list
        List of dates for predictions
    symbol : str
        Stock symbol
    save : bool
        Whether to save the plot
    """
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
        today = datetime.now().strftime('%Y%m%d')
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{symbol}_predictions_{today}.png"
        plt.savefig(filename, dpi=300)
        logger.info(f"Plot saved as {filename}")
    
    plt.show()


def save_to_csv(predictions, dates, symbol):
    """
    Save predictions to a CSV file.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary mapping model names to prediction arrays
    dates : list
        List of dates for predictions
    symbol : str
        Stock symbol
    """
    try:
        # Create DataFrame
        df = pd.DataFrame({'Date': [date.strftime('%Y-%m-%d') for date in dates]})
        
        for model, preds in predictions.items():
            df[model] = preds
        
        # Create output directory
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        today = datetime.now().strftime('%Y%m%d')
        filename = f"{output_dir}/{symbol}_predictions_{today}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Predictions saved to {filename}")
    
    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {str(e)}")


def run_backtest(models_dict, symbol):
    """
    Run backtesting for each model.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary mapping model names to model instances
    symbol : str
        Stock symbol
    """
    print("\n" + "="*50)
    print(f"{symbol} Backtesting Results")
    print("="*50)
    
    for model_name, model in models_dict.items():
        try:
            print(f"\nBacktesting {model_name} model...")
            if hasattr(model, 'backtest'):
                results = model.backtest(test_periods=3)
                
                # Display backtest results
                print(f"Average RMSE: {results['rmse'].mean():.4f}")
                print(f"Average MAE: {results['mae'].mean():.4f}")
                print(f"Average R²: {results['r2'].mean():.4f}")
            else:
                print(f"Backtesting not implemented for {model_name} model")
        
        except Exception as e:
            print(f"Error in backtesting {model_name}: {str(e)}")
    
    print("="*50 + "\n")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Display banner
    print("\n" + "="*60)
    print(f"Stock Price Prediction Tool")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Models: {args.models}")
    print(f"Days to predict: {args.days}")
    print(f"Training period: {args.period}")
    if args.train:
        print(f"Training mode: Force new training")
    if args.tune:
        print(f"Hyperparameter tuning: Enabled ({args.trials} trials)")
    print("="*60 + "\n")
    
    # Determine which models to use
    models_to_use = args.models.lower().split(',')
    if 'all' in models_to_use:
        models_to_use = ['xgboost', 'lstm', 'transformer']
    
    # Train or load models
    models_dict = {}
    for model_type in models_to_use:
        model = train_model(
            model_type=model_type,
            symbol=args.symbol,
            period=args.period,
            test_size=args.test_size,
            force_train=args.train,
            tune=args.tune,
            n_trials=args.trials
        )
        
        if model is not None:
            models_dict[model_type.capitalize()] = model
    
    if not models_dict:
        logger.error("No models were successfully loaded or trained. Exiting.")
        sys.exit(1)
    
    # Make predictions
    predictions, dates = make_predictions(models_dict, args.days)
    
    # Display predictions
    display_predictions(predictions, dates, args.symbol)
    
    # Run backtesting if requested
    if args.backtest:
        run_backtest(models_dict, args.symbol)
    
    # Save predictions to CSV if requested
    if args.save_csv:
        save_to_csv(predictions, dates, args.symbol)
    
    # Plot predictions if requested
    if args.plot:
        plot_predictions(predictions, dates, args.symbol, save=args.save_plot)
    
    logger.info("Prediction process completed successfully!")


if __name__ == "__main__":
    main()
