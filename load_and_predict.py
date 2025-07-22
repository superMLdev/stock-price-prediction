import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from algorithms.XGBoost import StockPricePredictor

# Create a new script that loads the saved model and makes predictions

def main():
    print("Loading saved XGBoost model...")
    
    # Initialize the predictor 
    predictor = StockPricePredictor(
        symbol="AAPL",
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    # Load the saved model/pipeline
    try:
        print("Attempting to load model from saved pipeline...")
        pipeline = predictor.load_model(
            load_pipeline=True,
            model_dir='models',
            filename='AAPL_pipeline.joblib'
        )
        print("Pipeline loaded successfully")
        
        # Make predictions using the loaded model
        print("\nMaking predictions with loaded model:")
        future_predictions = predictor.predict_future(
            days=5,
            use_cached_data=True,
            cache_dir='data'
        )
        print("\nPredictions for the next 5 trading days:")
        print(future_predictions)
        
        # Print some model metrics from the loaded pipeline
        print("\nLoaded Model Metrics:")
        for metric_name, metric_value in predictor.metrics.items():
            print(f"{metric_name}: {metric_value}")
            
    except Exception as e:
        print(f"Error loading or using model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
