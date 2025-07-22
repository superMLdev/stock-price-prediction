import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from algorithms.XGBoost import StockPricePredictor
from algorithms.Transformer import StockPriceTransformer
from algorithms.LSTM import LSTMStockPredictor

# Create a script that compares all three models: XGBoost, Transformer, and LSTM

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
    print("Note: Transformer model loading is not implemented, using results from previous run")
    
    # Here we just create a Transformer predictor to make new predictions
    predictor = StockPriceTransformer(
        symbol="AAPL",
        period="5y"
    )
    
    return predictor

def load_lstm_model():
    print("\nLoading LSTM model...")
    predictor = LSTMStockPredictor(
        symbol="AAPL",
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    try:
        predictor.load_model(
            model_path='models/lstm_stock_model.h5',
            load_pipeline=True,
            pipeline_path='models/AAPL_lstm_pipeline.joblib'
        )
        print("LSTM model loaded successfully")
        return predictor
    except Exception as e:
        print(f"Error loading LSTM model: {str(e)} - Model might not be trained yet")
        return None

def train_lstm_model():
    print("\nTraining LSTM model...")
    predictor = LSTMStockPredictor(
        symbol="AAPL",
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    try:
        # Run the pipeline with reduced epochs for demo
        model, metrics = predictor.run_pipeline(
            window_size=60,
            epochs=20,  # Reduced for demo
            batch_size=32,
            patience=5,
            use_cached_data=True,
            cache_dir='data'
        )
        print("LSTM model trained successfully")
        return predictor
    except Exception as e:
        print(f"Error training LSTM model: {str(e)}")
        return None

def compare_predictions(xgb_predictor, transformer_metrics, lstm_predictor):
    print("\nGenerating predictions from models...")
    
    # Get predictions from XGBoost
    xgb_predictions = None
    if xgb_predictor:
        xgb_predictions = xgb_predictor.predict_future(
            days=5,
            use_cached_data=True,
            cache_dir='data'
        )
        print("\nXGBoost predictions:")
        print(xgb_predictions)
    
    # Get predictions from LSTM
    lstm_predictions = None
    if lstm_predictor:
        lstm_predictions = lstm_predictor.predict_future(
            days=5,
            use_cached_data=True,
            cache_dir='data'
        )
        print("\nLSTM predictions:")
        print(lstm_predictions)
    
    # Compare metrics
    print("\nModel Performance Comparison:")
    
    # Create a metrics table
    metrics_data = []
    
    # XGBoost metrics
    if xgb_predictor and hasattr(xgb_predictor, 'metrics') and xgb_predictor.metrics:
        metrics_data.append({
            'Model': 'XGBoost',
            'Test RMSE': xgb_predictor.metrics['test_rmse'],
            'Test MAE': xgb_predictor.metrics['test_mae'],
            'Test MAPE': xgb_predictor.metrics['test_mape'],
            'Test R²': xgb_predictor.metrics['test_r2']
        })
        print("XGBoost Metrics:")
        print(f"Test RMSE: {xgb_predictor.metrics['test_rmse']:.4f}")
        print(f"Test MAE: {xgb_predictor.metrics['test_mae']:.4f}")
        print(f"Test MAPE: {xgb_predictor.metrics['test_mape']:.4f}%")
        print(f"Test R²: {xgb_predictor.metrics['test_r2']:.4f}")
    
    # Transformer metrics (from previous run)
    metrics_data.append({
        'Model': 'Transformer',
        'Test RMSE': 15.0221,
        'Test MAE': 12.9198,
        'Test MAPE': 9.76,
        'Test R²': 0.3710
    })
    print("\nTransformer Metrics (from previous run):")
    print("Test RMSE: 15.0221")
    print("Test MAE: 12.9198")
    print("Test MAPE: 9.76%")
    print("Test R²: 0.3710")
    
    # LSTM metrics
    if lstm_predictor and hasattr(lstm_predictor, 'metrics') and lstm_predictor.metrics:
        metrics_data.append({
            'Model': 'LSTM',
            'Test RMSE': lstm_predictor.metrics['test_rmse'],
            'Test MAE': lstm_predictor.metrics['test_mae'],
            'Test MAPE': lstm_predictor.metrics['test_mape'],
            'Test R²': lstm_predictor.metrics['test_r2']
        })
        print("\nLSTM Metrics:")
        print(f"Test RMSE: {lstm_predictor.metrics['test_rmse']:.4f}")
        print(f"Test MAE: {lstm_predictor.metrics['test_mae']:.4f}")
        print(f"Test MAPE: {lstm_predictor.metrics['test_mape']:.4f}%")
        print(f"Test R²: {lstm_predictor.metrics['test_r2']:.4f}")
    
    # Create comparison visualizations
    create_comparison_chart(metrics_data)
    
    if xgb_predictions is not None and lstm_predictions is not None:
        compare_prediction_charts(xgb_predictions, lstm_predictions)

def create_comparison_chart(metrics_data):
    # Create a bar chart comparing the metrics
    if not metrics_data:
        print("No metrics data available for comparison chart")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.set_index('Model')
    
    # Plot RMSE
    ax1 = plt.subplot(2, 2, 1)
    metrics_df['Test RMSE'].plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title('Root Mean Squared Error (RMSE)', fontsize=14)
    ax1.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_df['Test RMSE']):
        ax1.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=10)
    
    # Plot MAE
    ax2 = plt.subplot(2, 2, 2)
    metrics_df['Test MAE'].plot(kind='bar', color='lightgreen', ax=ax2)
    ax2.set_title('Mean Absolute Error (MAE)', fontsize=14)
    ax2.set_ylabel('MAE (lower is better)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_df['Test MAE']):
        ax2.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=10)
    
    # Plot MAPE
    ax3 = plt.subplot(2, 2, 3)
    metrics_df['Test MAPE'].plot(kind='bar', color='salmon', ax=ax3)
    ax3.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14)
    ax3.set_ylabel('MAPE % (lower is better)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_df['Test MAPE']):
        ax3.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=10)
    
    # Plot R²
    ax4 = plt.subplot(2, 2, 4)
    metrics_df['Test R²'].plot(kind='bar', color='mediumpurple', ax=ax4)
    ax4.set_title('R-squared (R²)', fontsize=14)
    ax4.set_ylabel('R² (higher is better)', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(metrics_df['Test R²']):
        ax4.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    
    # Find best model based on RMSE
    best_model = metrics_df['Test RMSE'].idxmin()
    
    plt.suptitle('Stock Price Prediction Model Comparison', fontsize=16)
    plt.figtext(0.5, 0.01, 
               f"Based on RMSE, {best_model} performs best for AAPL stock prediction.", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model_comparison_all.png')
    print("\nComparison chart saved to 'model_comparison_all.png'")
    plt.close()

def compare_prediction_charts(xgb_predictions, lstm_predictions):
    """
    Create a chart comparing the predictions from XGBoost and LSTM models.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot XGBoost predictions
    plt.plot(xgb_predictions.index, xgb_predictions['Predicted_Close'], 
             label='XGBoost Predictions', marker='o', color='blue', linewidth=2)
    
    # Plot LSTM predictions
    plt.plot(lstm_predictions.index, lstm_predictions['Predicted_Close'], 
             label='LSTM Predictions', marker='s', color='red', linewidth=2)
    
    plt.title('AAPL Price Predictions Comparison', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Predicted Close Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotations for the last prediction
    plt.annotate(f'${xgb_predictions["Predicted_Close"].iloc[-1]:.2f}',
                xy=(xgb_predictions.index[-1], xgb_predictions['Predicted_Close'].iloc[-1]),
                xytext=(10, 10), textcoords='offset points', fontsize=10)
    
    plt.annotate(f'${lstm_predictions["Predicted_Close"].iloc[-1]:.2f}',
                xy=(lstm_predictions.index[-1], lstm_predictions['Predicted_Close'].iloc[-1]),
                xytext=(10, -15), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    print("Prediction comparison chart saved to 'prediction_comparison.png'")
    plt.close()

def main():
    print("\n=== Stock Price Prediction Model Comparison ===\n")
    
    # Load models
    xgb_predictor = load_xgboost_model()
    transformer_metrics = {'test_rmse': 15.0221, 'test_mae': 12.9198, 'test_mape': 9.76, 'test_r2': 0.3710}
    
    # Try to load LSTM model, if not available, train it
    lstm_predictor = load_lstm_model()
    if lstm_predictor is None:
        print("LSTM model not found, training now...")
        lstm_predictor = train_lstm_model()
    
    # Compare predictions and metrics
    compare_predictions(xgb_predictor, transformer_metrics, lstm_predictor)
    
    # Conclusion
    print("\nConclusion:")
    if xgb_predictor and lstm_predictor:
        xgb_rmse = xgb_predictor.metrics['test_rmse'] if hasattr(xgb_predictor, 'metrics') and xgb_predictor.metrics else float('inf')
        lstm_rmse = lstm_predictor.metrics['test_rmse'] if hasattr(lstm_predictor, 'metrics') and lstm_predictor.metrics else float('inf')
        
        if xgb_rmse < lstm_rmse and xgb_rmse < 15.0221:
            print("The XGBoost model performs best for AAPL stock prediction based on the test RMSE.")
            print(f"XGBoost RMSE: {xgb_rmse:.4f} vs LSTM RMSE: {lstm_rmse:.4f} vs Transformer RMSE: 15.0221")
            print("\nRecommendation: Use the XGBoost model for stock price prediction.")
        elif lstm_rmse < xgb_rmse and lstm_rmse < 15.0221:
            print("The LSTM model performs best for AAPL stock prediction based on the test RMSE.")
            print(f"LSTM RMSE: {lstm_rmse:.4f} vs XGBoost RMSE: {xgb_rmse:.4f} vs Transformer RMSE: 15.0221")
            print("\nRecommendation: Use the LSTM model for stock price prediction.")
        else:
            print("The Transformer model performs best for AAPL stock prediction based on the test RMSE.")
            print(f"Transformer RMSE: 15.0221 vs XGBoost RMSE: {xgb_rmse:.4f} vs LSTM RMSE: {lstm_rmse:.4f}")
            print("\nRecommendation: Use the Transformer model for stock price prediction.")
    else:
        print("Unable to make a complete comparison as some models are unavailable.")

if __name__ == "__main__":
    main()
