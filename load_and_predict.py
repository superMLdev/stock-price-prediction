import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse
import logging
import traceback
from datetime import datetime
from algorithms.XGBoost import StockPricePredictor
from algorithms.LSTM import LSTMStockPredictor
from algorithms.Transformer import StockPriceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def predict_with_xgboost(symbol, days=5, use_cached=True, cache_dir='data', model_dir='models'):
    """Make predictions with a pre-trained XGBoost model"""
    logger.info(f"Loading saved XGBoost model for {symbol}...")
    
    # Initialize the predictor 
    predictor = StockPricePredictor(
        symbol=symbol,
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    # Load the saved model/pipeline
    try:
        pipeline_file = f"{symbol}_pipeline.joblib"
        if not os.path.exists(os.path.join(model_dir, pipeline_file)):
            pipeline_file = "xgboost_stock_model.joblib"
            
        logger.info(f"Attempting to load model from {pipeline_file}...")
        pipeline = predictor.load_model(
            load_pipeline=True,
            model_dir=model_dir,
            filename=pipeline_file
        )
        logger.info("Model loaded successfully")
        
        # Make predictions using the loaded model
        logger.info(f"Making predictions for {symbol} for the next {days} trading days...")
        future_predictions = predictor.predict_future(
            days=days,
            use_cached_data=use_cached,
            cache_dir=cache_dir
        )
        
        # Save predictions to CSV
        predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        today = datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(predictions_dir, f"{symbol}_predictions_{today}.csv")
        
        # Get future dates
        future_dates = predictor.get_future_dates(days)
        
        # Create dataframe with predictions
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_predictions
        })
        
        # Save to CSV
        pred_df.to_csv(csv_path, index=False)
        logger.info(f"Predictions saved to {csv_path}")
        
        # Display predictions
        logger.info("\nPredictions for the next trading days:")
        for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
            logger.info(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
            
        return future_predictions, future_dates
        
    except Exception as e:
        logger.error(f"Error loading or using model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def predict_with_lstm(symbol, days=5, use_cached=True, cache_dir='data', model_dir='models'):
    """Make predictions with a pre-trained LSTM model"""
    logger.info(f"Loading saved LSTM model for {symbol}...")
    
    try:
        # Initialize the predictor
        predictor = LSTMStockPredictor(
            symbol=symbol,
            period="5y",
            test_size=0.2
        )
        
        # Load the saved model
        pipeline_file = f"{symbol}_lstm_pipeline.joblib"
        if not os.path.exists(os.path.join(model_dir, pipeline_file)):
            pipeline_file = "lstm_stock_model.joblib"
            
        logger.info(f"Attempting to load model from {pipeline_file}...")
        predictor.load_model(load_pipeline=True, model_dir=model_dir, filename=pipeline_file)
        
        # Make predictions
        logger.info(f"Making predictions for {symbol} for the next {days} trading days...")
        predictions = predictor.predict_future(days=days, use_cached=use_cached)
        
        # Get future dates
        future_dates = predictor.get_future_dates(days)
        
        # Display predictions
        logger.info("\nLSTM Predictions for the next trading days:")
        for i, (date, price) in enumerate(zip(future_dates, predictions)):
            logger.info(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
            
        return predictions, future_dates
        
    except Exception as e:
        logger.error(f"Error loading or using LSTM model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def predict_with_transformer(symbol, days=5, use_cached=True, cache_dir='data', model_dir='transformer_model'):
    """Make predictions with a pre-trained Transformer model"""
    logger.info(f"Loading saved Transformer model for {symbol}...")
    
    try:
        # Initialize the transformer
        transformer = StockPriceTransformer(
            symbol=symbol,
            period="5y"
        )
        
        # Load the model
        logger.info("Attempting to load Transformer model...")
        transformer.load_model(model_dir=model_dir)
        
        # Make predictions
        logger.info(f"Making predictions for {symbol} for the next {days} trading days...")
        predictions = transformer.predict_future(days=days, use_cached=use_cached)
        
        # Get future dates
        future_dates = transformer.get_future_dates(days)
        
        # Display predictions
        logger.info("\nTransformer Predictions for the next trading days:")
        for i, (date, price) in enumerate(zip(future_dates, predictions)):
            logger.info(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
            
        return predictions, future_dates
        
    except Exception as e:
        logger.error(f"Error loading or using Transformer model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Load a pre-trained model and make stock price predictions")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol (e.g., AAPL, MSFT)")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lstm", "transformer"], 
                        help="Model to use for prediction")
    parser.add_argument("--days", type=int, default=5, help="Number of days to predict")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached stock data")
    
    args = parser.parse_args()
    
    # Make predictions with the selected model
    if args.model.lower() == "xgboost":
        predict_with_xgboost(args.symbol, args.days, not args.no_cache)
    elif args.model.lower() == "lstm":
        predict_with_lstm(args.symbol, args.days, not args.no_cache)
    elif args.model.lower() == "transformer":
        predict_with_transformer(args.symbol, args.days, not args.no_cache)
    else:
        logger.error(f"Unknown model: {args.model}")

if __name__ == "__main__":
    main()
