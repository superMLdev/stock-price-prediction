import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from algorithms.XGBoost import StockPricePredictor
from algorithms.Transformer import StockPriceTransformer

# Create a script that compares the XGBoost and Transformer models

def load_xgboost_model():
    print("Loading XGBoost model...")
    predictor = StockPricePredictor(
        symbol="AAPL",
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    try:
        pipeline = predictor.load_model(
            load_pipeline=True,
            model_dir='models',
            filename='AAPL_pipeline.joblib'
        )
        print("XGBoost model loaded successfully")
        return predictor
    except Exception as e:
        print(f"Error loading XGBoost model: {str(e)}")
        return None

def load_transformer_model():
    print("\nLoading Transformer model...")
    # Since we don't have a load_model method in Transformer yet, we'll just note this
    print("Note: Transformer model loading is not implemented, using results from previous run")
    
    # Here we just create a Transformer predictor to make new predictions
    predictor = StockPriceTransformer(
        symbol="AAPL",
        period="5y"
    )
    
    return predictor

def compare_predictions(xgb_predictor, transformer_predictor):
    print("\nGenerating predictions from both models...")
    
    # Get predictions from XGBoost
    xgb_predictions = xgb_predictor.predict_future(
        days=5,
        use_cached_data=True,
        cache_dir='data'
    )
    
    # For transformer, we would need to run the pipeline again 
    # In a real implementation, you would load the saved transformer model
    print("\nXGBoost predictions:")
    print(xgb_predictions)
    
    # Compare metrics
    print("\nModel Performance Comparison:")
    print("XGBoost Metrics:")
    print(f"Test RMSE: {xgb_predictor.metrics['test_rmse']:.4f}")
    print(f"Test MAE: {xgb_predictor.metrics['test_mae']:.4f}")
    print(f"Test MAPE: {xgb_predictor.metrics['test_mape']:.4f}%")
    print(f"Test R²: {xgb_predictor.metrics['test_r2']:.4f}")
    
    print("\nTransformer Metrics (from previous run):")
    # These are from our previous run
    print("Test RMSE: 15.0221")
    print("Test MAE: 12.9198")
    print("Test MAPE: 9.76%")
    print("Test R²: 0.3710")
    
    # Create a performance comparison visualization
    create_comparison_chart(xgb_predictor)

def create_comparison_chart(xgb_predictor):
    # Create a bar chart comparing the metrics
    plt.figure(figsize=(12, 8))
    
    # Data for the chart
    models = ['XGBoost', 'Transformer']
    rmse_values = [xgb_predictor.metrics['test_rmse'], 15.0221]
    mae_values = [xgb_predictor.metrics['test_mae'], 12.9198]
    r2_values = [xgb_predictor.metrics['test_r2'], 0.3710]
    
    # Set up bar positions
    bar_width = 0.25
    r1 = np.arange(len(models))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create the grouped bar chart
    plt.bar(r1, rmse_values, width=bar_width, label='RMSE', color='skyblue')
    plt.bar(r2, mae_values, width=bar_width, label='MAE', color='lightgreen')
    plt.bar(r3, r2_values, width=bar_width, label='R²', color='salmon')
    
    # Add labels and title
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xticks([r + bar_width for r in range(len(models))], models)
    plt.legend()
    
    # Add a summary text
    if xgb_predictor.metrics['test_rmse'] < 15.0221:
        better_model = "XGBoost"
    else:
        better_model = "Transformer"
    
    plt.figtext(0.5, 0.01, 
                f"Summary: {better_model} performs better for AAPL stock prediction based on RMSE.", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save the chart
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model_comparison.png')
    print("\nComparison chart saved to 'model_comparison.png'")
    plt.close()

def main():
    print("\n=== Stock Price Prediction Model Comparison ===\n")
    
    # Load models
    xgb_predictor = load_xgboost_model()
    transformer_predictor = load_transformer_model()
    
    if xgb_predictor:
        # Compare predictions and metrics
        compare_predictions(xgb_predictor, transformer_predictor)
        
        # Conclusion
        print("\nConclusion:")
        if xgb_predictor.metrics['test_rmse'] < 15.0221:
            print("The XGBoost model performs better for AAPL stock prediction based on the test RMSE.")
            print(f"XGBoost RMSE: {xgb_predictor.metrics['test_rmse']:.4f} vs Transformer RMSE: 15.0221")
            print("\nRecommendation: Use the XGBoost model for stock price prediction.")
        else:
            print("The Transformer model performs better for AAPL stock prediction based on the test RMSE.")
            print(f"Transformer RMSE: 15.0221 vs XGBoost RMSE: {xgb_predictor.metrics['test_rmse']:.4f}")
            print("\nRecommendation: Use the Transformer model for stock price prediction.")

if __name__ == "__main__":
    main()
