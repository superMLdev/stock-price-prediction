import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lstm_stock_model.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class LSTMStockPredictor:
    """
    End-to-end pipeline for predicting stock prices using LSTM
    
    This class handles:
    - Data fetching from Yahoo Finance
    - Feature engineering
    - Data preprocessing
    - LSTM model training
    - Evaluation and visualization
    - Model persistence
    """
    
    def __init__(self, symbol, period="5y", test_size=0.2, target_col='Close', prediction_horizon=1):
        """
        Initialize the stock price prediction pipeline.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        period : str
            Data period to fetch (e.g., '5y' for 5 years)
        test_size : float
            Proportion of data to use for testing (0-1)
        target_col : str
            Column to predict (usually 'Close')
        prediction_horizon : int
            Number of days ahead to predict (default: 1 for next day)
        """
        self.symbol = symbol
        self.period = period
        self.test_size = test_size
        self.target_col = target_col
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.data = None
        self.featured_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.metrics = None
        self.window_size = 60  # Default window size for time series

    def fetch_data(self, use_cached=True, cache_dir='data'):
        """
        Fetch historical stock data from Yahoo Finance or from local cache if available.
        
        Parameters:
        -----------
        use_cached : bool
            Whether to use cached data if available
        cache_dir : str
            Directory to store/retrieve cached data
            
        Returns:
        --------
        pandas.DataFrame
            Raw OHLCV data
        """
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Construct cache file path
            cache_file = os.path.join(cache_dir, f"{self.symbol}_{self.period}_data.csv")
            
            # Check if cached data exists and is requested
            if os.path.exists(cache_file) and use_cached:
                logger.info(f"Loading cached data for {self.symbol} from {cache_file}")
                self.data = pd.read_csv(cache_file, index_col=0)
                # Convert index to datetime properly handling timezone issues
                self.data.index = pd.DatetimeIndex(pd.to_datetime(self.data.index, utc=True).tz_localize(None))
                logger.info(f"Loaded {len(self.data)} rows of data from cache")
            else:
                # Download data from Yahoo Finance
                logger.info(f"Fetching data for {self.symbol} from Yahoo Finance...")
                stock = yf.Ticker(self.symbol)
                self.data = stock.history(period=self.period)

                # Check if we got any data
                if self.data.empty:
                    raise ValueError(f"No data returned for symbol {self.symbol}")

                # Handle missing values in the raw data
                if self.data.isnull().sum().sum() > 0:
                    logger.warning(f"Found {self.data.isnull().sum().sum()} missing values in raw data")
                    self.data = self.data.fillna(method='ffill')

                # Save data to cache
                logger.info(f"Saving {len(self.data)} rows of data to {cache_file}")
                self.data.to_csv(cache_file)
                logger.info(f"Downloaded {len(self.data)} rows of data")
            
            return self.data

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with OHLCV data
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with added technical indicators
        """
        # Moving Average Convergence Divergence
        macd = MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Relative Strength Index
        rsi = RSIIndicator(close=df['Close'])
        df['rsi'] = rsi.rsi()

        # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'])
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()
        df['bollinger_width'] = bollinger.bollinger_wband()
        df['bollinger_pband'] = bollinger.bollinger_pband()

        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            sma = SMAIndicator(close=df['Close'], window=window)
            ema = EMAIndicator(close=df['Close'], window=window)
            df[f'sma_{window}'] = sma.sma_indicator()
            df[f'ema_{window}'] = ema.ema_indicator()

            # Price distance from moving averages (%)
            df[f'close_sma_{window}_ratio'] = df['Close'] / df[f'sma_{window}'] - 1
            df[f'close_ema_{window}_ratio'] = df['Close'] / df[f'ema_{window}'] - 1

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Average True Range (volatility)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])
        df['atr'] = atr.average_true_range()

        # Volume Indicators
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
        df['obv'] = obv.on_balance_volume()

        # VWAP (if intraday data, otherwise will use adjusted close)
        try:
            vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'],
                                            close=df['Close'], volume=df['Volume'])
            df['vwap'] = vwap.volume_weighted_average_price()
        except:
            logger.warning("Couldn't calculate VWAP, possibly due to data frequency")

        return df

    def engineer_features(self, lookback_days=30):
        """
        Create features for LSTM training.
        
        Parameters:
        -----------
        lookback_days : int
            Number of past days to use for lag features
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with engineered features
        """
        try:
            logger.info("Engineering features...")

            if self.data is None:
                self.fetch_data()

            df = self.data.copy()

            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Price-based features
            for i in range(1, lookback_days + 1):
                # Lag features - more selective for LSTM to avoid dimensionality issues
                if i <= 5:  # Only use a few lag features for key columns
                    for col in ['Close', 'Volume', 'rsi', 'macd']:
                        df[f'{col.lower()}_lag_{i}'] = df[col].shift(i)

                # Return-based features
                if i <= 10:  # Only create return features for shorter periods
                    df[f'return_{i}d'] = df['Close'].pct_change(i)
                    df[f'volume_change_{i}d'] = df['Volume'].pct_change(i)

            # Price gaps
            df['gap_open'] = df['Open'] / df['Close'].shift(1) - 1
            df['gap_close'] = df['Close'] / df['Open'] - 1

            # High-Low range
            df['day_range'] = (df['High'] - df['Low']) / df['Open']

            # Price position within day's range
            df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

            # Volatility measures
            for window in [5, 10, 20]:
                df[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window=window).std()

                # Price momentum
                df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1

            # Target variable (next day's closing price)
            df[f'target_{self.prediction_horizon}d'] = df[self.target_col].shift(-self.prediction_horizon)

            # Drop rows with NaN values
            self.featured_data = df.dropna()

            # Log feature creation
            initial_cols = len(self.data.columns)
            final_cols = len(self.featured_data.columns)
            logger.info(f"Created {final_cols - initial_cols} new features, total features: {final_cols}")

            return self.featured_data

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def create_sequences(self, X, y, window_size=None):
        """
        Create sequences for LSTM input from time series data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature data
        y : numpy.ndarray
            Target data
        window_size : int, optional
            Size of the time window, defaults to self.window_size
            
        Returns:
        --------
        tuple
            (X_seq, y_seq) - Sequences for LSTM
        """
        if window_size is None:
            window_size = self.window_size
            
        X_seq, y_seq = [], []
        for i in range(len(X) - window_size):
            X_seq.append(X[i:i + window_size])
            y_seq.append(y[i + window_size])
            
        return np.array(X_seq), np.array(y_seq)

    def preprocess_data(self, window_size=60, scale_features=True, scaler_type='minmax'):
        """
        Preprocess data for LSTM training, including scaling and sequence creation.
        
        Parameters:
        -----------
        window_size : int
            Size of the time window for LSTM
        scale_features : bool
            Whether to scale the features
        scaler_type : str
            Type of scaler to use ('standard', 'robust', or 'minmax')
            
        Returns:
        --------
        tuple
            X_train_seq, X_test_seq, y_train, y_test
        """
        try:
            logger.info("Preprocessing data...")
            self.window_size = window_size

            if self.featured_data is None:
                self.engineer_features()

            # Prepare features and target
            target_col = f'target_{self.prediction_horizon}d'
            feature_columns = [col for col in self.featured_data.columns
                              if col not in [target_col, 'Dividends', 'Stock Splits']]

            self.X = self.featured_data[feature_columns].values
            self.y = self.featured_data[target_col].values.reshape(-1, 1)

            # Scale features
            if scale_features:
                if scaler_type == 'standard':
                    self.scaler = StandardScaler()
                elif scaler_type == 'robust':
                    self.scaler = RobustScaler()
                else:  # minmax is often better for neural networks
                    self.scaler = MinMaxScaler(feature_range=(0, 1))
                
                self.X = self.scaler.fit_transform(self.X)
                
                # Create a separate scaler for the target variable
                self.target_scaler = MinMaxScaler(feature_range=(0, 1))
                self.y = self.target_scaler.fit_transform(self.y)

            # Create sequences for LSTM
            X_seq, y_seq = self.create_sequences(self.X, self.y, window_size)
            
            # Split data (time-based split for time series)
            split_idx = int(len(X_seq) * (1 - self.test_size))
            
            self.X_train = X_seq[:split_idx]
            self.X_test = X_seq[split_idx:]
            self.y_train = y_seq[:split_idx]
            self.y_test = y_seq[split_idx:]
            
            # Save the original test dates for plotting
            test_dates_idx = np.arange(window_size, len(self.X)) - (len(self.X) - len(self.X_test))
            test_dates_idx = test_dates_idx[test_dates_idx >= 0]
            self.test_dates = self.featured_data.index[test_dates_idx]

            logger.info(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            logger.info(f"Sequence shape: {self.X_train.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def build_model(self, input_shape):
        """
        Build LSTM model architecture.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (window_size, n_features)
            
        Returns:
        --------
        tensorflow.keras.Model
            Compiled LSTM model
        """
        try:
            logger.info(f"Building LSTM model with input shape {input_shape}...")
            
            model = keras.Sequential([
                layers.LSTM(100, return_sequences=True, input_shape=input_shape),
                layers.Dropout(0.2),
                layers.LSTM(50, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(25, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae', 'mse']
            )
            
            model.summary(print_fn=logger.info)
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train_model(self, epochs=50, batch_size=32, patience=10):
        """
        Train LSTM model with early stopping.
        
        Parameters:
        -----------
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        tensorflow.keras.Model
            Trained LSTM model
        """
        try:
            logger.info(f"Training LSTM model for {epochs} epochs (max)...")
            
            if self.model is None:
                self.model = self.build_model(input_shape=(self.X_train.shape[1], self.X_train.shape[2]))
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # Train the model
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self.plot_training_history(history)
            
            logger.info("LSTM model training completed")
            
            return self.model, history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def plot_training_history(self, history):
        """
        Plot the training and validation loss.
        
        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.savefig('lstm_training_history.png')
            plt.close()
            
            logger.info("Training history plot saved to 'lstm_training_history.png'")
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")

    def evaluate_model(self):
        """
        Evaluate model performance.
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        try:
            logger.info("Evaluating model performance...")

            # Make predictions
            self.y_pred_train = self.model.predict(self.X_train)
            self.y_pred_test = self.model.predict(self.X_test)
            
            # Inverse transform if data was scaled
            if hasattr(self, 'target_scaler'):
                self.y_train_orig = self.target_scaler.inverse_transform(self.y_train.reshape(-1, 1)).flatten()
                self.y_test_orig = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
                self.y_pred_train = self.target_scaler.inverse_transform(self.y_pred_train).flatten()
                self.y_pred_test = self.target_scaler.inverse_transform(self.y_pred_test).flatten()
            else:
                self.y_train_orig = self.y_train
                self.y_test_orig = self.y_test

            # Calculate metrics
            metrics = {}

            # Training metrics
            metrics['train_rmse'] = np.sqrt(mean_squared_error(self.y_train_orig, self.y_pred_train))
            metrics['train_mae'] = mean_absolute_error(self.y_train_orig, self.y_pred_train)
            metrics['train_r2'] = r2_score(self.y_train_orig, self.y_pred_train)

            # Testing metrics
            metrics['test_rmse'] = np.sqrt(mean_squared_error(self.y_test_orig, self.y_pred_test))
            metrics['test_mae'] = mean_absolute_error(self.y_test_orig, self.y_pred_test)
            metrics['test_r2'] = r2_score(self.y_test_orig, self.y_pred_test)

            # Calculate MAPE (Mean Absolute Percentage Error)
            metrics['train_mape'] = np.mean(np.abs((self.y_train_orig - self.y_pred_train) / self.y_train_orig)) * 100
            metrics['test_mape'] = np.mean(np.abs((self.y_test_orig - self.y_pred_test) / self.y_test_orig)) * 100

            # Display metrics
            logger.info(f"\nModel Performance for {self.symbol}:")
            logger.info(f"Train RMSE: {metrics['train_rmse']:.4f}")
            logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
            logger.info(f"Train MAE: {metrics['train_mae']:.4f}")
            logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
            logger.info(f"Train MAPE: {metrics['train_mape']:.2f}%")
            logger.info(f"Test MAPE: {metrics['test_mape']:.2f}%")
            logger.info(f"Train R²: {metrics['train_r2']:.4f}")
            logger.info(f"Test R²: {metrics['test_r2']:.4f}")

            self.metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def visualize_results(self, save_path='lstm_visualization.png'):
        """
        Create visualizations for model results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save visualization figures
        """
        try:
            logger.info("Creating visualizations...")
            
            # Check if we have test_dates attribute, if not, create it
            if not hasattr(self, 'test_dates'):
                if hasattr(self, 'featured_data') and self.featured_data is not None:
                    test_size = int(len(self.featured_data) * self.test_size)
                    self.test_dates = self.featured_data.index[-test_size:]
                else:
                    # If we can't determine test_dates, use a range as fallback
                    self.test_dates = pd.date_range(end=datetime.now(), periods=len(self.y_test_orig))

            # Create a figure with subplots
            fig = plt.figure(figsize=(20, 15))

            # 1. Actual vs Predicted plot
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(self.test_dates, self.y_test_orig, label='Actual', color='blue', linewidth=2)
            ax1.plot(self.test_dates, self.y_pred_test, label='Predicted', color='red', linestyle='--', linewidth=2)
            ax1.set_title(f'{self.symbol} - LSTM: Actual vs Predicted Price', fontsize=14)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Stock Price', fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Add confidence bands
            residuals = self.y_test_orig - self.y_pred_test
            residual_std = np.std(residuals)

            # Plot confidence bands (±2 standard deviations)
            ax1.fill_between(self.test_dates,
                            self.y_pred_test - 2*residual_std,
                            self.y_pred_test + 2*residual_std,
                            color='red', alpha=0.2, label='95% Confidence Interval')

            # 2. Training history plot (if available)
            ax2 = plt.subplot(2, 2, 2)
            if hasattr(self, 'history'):
                ax2.plot(self.history.history['loss'], label='Training Loss', color='blue')
                ax2.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
                ax2.set_title('LSTM Model Loss', fontsize=14)
                ax2.set_ylabel('Loss (MSE)', fontsize=12)
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Training history not available', 
                        horizontalalignment='center', verticalalignment='center', fontsize=12)
                ax2.axis('off')

            # 3. Residuals plot
            ax3 = plt.subplot(2, 2, 3)
            ax3.scatter(self.y_pred_test, residuals, alpha=0.5, color='blue')
            ax3.axhline(y=0, color='r', linestyle='-')

            # Add trend line to residuals
            z = np.polyfit(self.y_pred_test, residuals, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(self.y_pred_test), p(sorted(self.y_pred_test)), "r--", alpha=0.8)

            ax3.set_title('Residuals Plot', fontsize=14)
            ax3.set_xlabel('Predicted Values', fontsize=12)
            ax3.set_ylabel('Residuals', fontsize=12)
            ax3.grid(True, alpha=0.3)

            # 4. Error over time
            ax4 = plt.subplot(2, 2, 4)
            abs_error = np.abs(self.y_test_orig - self.y_pred_test)
            error_df = pd.DataFrame({'Date': self.test_dates, 'Absolute Error': abs_error})

            # Plot error over time
            ax4.plot(error_df['Date'], error_df['Absolute Error'], color='blue', alpha=0.7)

            # Add trend line
            z = np.polyfit(range(len(error_df)), error_df['Absolute Error'], 1)
            p = np.poly1d(z)
            ax4.plot(error_df['Date'], p(range(len(error_df))), "r--", linewidth=2)

            ax4.set_title('Prediction Error Over Time', fontsize=14)
            ax4.set_ylabel('Absolute Error', fontsize=12)
            ax4.set_xlabel('Date', fontsize=12)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualizations saved to {save_path}")

            plt.show()

            # Additional plot for trading simulation
            self.plot_trading_simulation()

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
            raise

    def plot_trading_simulation(self):
        """
        Create a simple trading simulation plot based on predictions.
        """
        try:
            # Create a trading simulation plot
            plt.figure(figsize=(12, 6))

            # Create time series of actual prices
            actual_prices = pd.Series(self.y_test_orig, index=self.test_dates)
            predicted_prices = pd.Series(self.y_pred_test, index=self.test_dates)

            # Get actual price changes
            actual_returns = actual_prices.pct_change().fillna(0)

            # Get predicted price changes
            predicted_returns = predicted_prices.pct_change().fillna(0)

            # Simple strategy: Buy when predicted return is positive, sell when negative
            signal = np.sign(predicted_returns)

            # Calculate strategy returns
            strategy_returns = signal.shift(1) * actual_returns
            strategy_returns.fillna(0, inplace=True)

            # Calculate cumulative returns
            cum_actual_returns = (1 + actual_returns).cumprod() - 1
            cum_strategy_returns = (1 + strategy_returns).cumprod() - 1

            # Buy and hold vs Strategy plot
            plt.plot(cum_actual_returns, label='Buy & Hold', color='blue', linewidth=2)
            plt.plot(cum_strategy_returns, label='LSTM Strategy', color='green', linewidth=2)

            # Add annotations
            final_bh_return = cum_actual_returns.iloc[-1]
            final_strategy_return = cum_strategy_returns.iloc[-1]

            plt.scatter(cum_actual_returns.index[-1], final_bh_return, color='blue', zorder=5)
            plt.scatter(cum_strategy_returns.index[-1], final_strategy_return, color='green', zorder=5)

            plt.annotate(f'{final_bh_return:.2%}',
                        (cum_actual_returns.index[-1], final_bh_return),
                        xytext=(10, 10), textcoords='offset points', fontsize=12)

            plt.annotate(f'{final_strategy_return:.2%}',
                        (cum_strategy_returns.index[-1], final_strategy_return),
                        xytext=(10, -15), textcoords='offset points', fontsize=12)

            plt.title(f'{self.symbol} - LSTM Trading Simulation (Test Period)', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create visualization directory if needed
            os.makedirs('visualizations', exist_ok=True)
            save_path = os.path.join('visualizations', f'lstm_trading_simulation_{self.symbol}.png')
            plt.savefig(save_path)
            logger.info(f"Trading simulation plot saved to {save_path}")
            
            plt.show()

            # Calculate some trading statistics
            n_trades = np.sum(np.abs(signal.diff().fillna(0)) > 0)
            win_rate = np.mean(strategy_returns > 0)
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()

            logger.info(f"\nTrading Simulation Results:")
            logger.info(f"Buy & Hold Return: {final_bh_return:.2%}")
            logger.info(f"Strategy Return: {final_strategy_return:.2%}")
            logger.info(f"Number of Trades: {n_trades}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe:.2f}")

        except Exception as e:
            logger.error(f"Error in trading simulation plot: {str(e)}")

    def save_model(self, model_path='models/lstm_stock_model.h5', save_pipeline=True, pipeline_path='models/lstm_pipeline.joblib'):
        """
        Save the trained model and optionally the full pipeline.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        save_pipeline : bool
            Whether to save the full pipeline including scaler
        pipeline_path : str
            Path to save the pipeline
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save Keras model
            self.model.save(model_path)
            logger.info(f"Model saved to '{model_path}'")
            
            if save_pipeline:
                # Save the full pipeline
                pipeline_data = {
                    'scaler': self.scaler,
                    'target_scaler': self.target_scaler,
                    'metrics': self.metrics,
                    'symbol': self.symbol,
                    'target_col': self.target_col,
                    'prediction_horizon': self.prediction_horizon,
                    'window_size': self.window_size
                }
                
                joblib.dump(pipeline_data, pipeline_path)
                logger.info(f"Pipeline saved to '{pipeline_path}'")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path='models/lstm_stock_model.h5', load_pipeline=False, pipeline_path='models/lstm_pipeline.joblib'):
        """
        Load a trained model and optionally the full pipeline.
        
        Parameters:
        -----------
        model_path : str
            Path to load the model from
        load_pipeline : bool
            Whether to load the full pipeline
        pipeline_path : str
            Path to load the pipeline from
            
        Returns:
        --------
        tensorflow.keras.Model
            Loaded model
        """
        try:
            # Load Keras model
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from '{model_path}'")
            
            if load_pipeline:
                # Load the pipeline
                pipeline_data = joblib.load(pipeline_path)
                
                self.scaler = pipeline_data['scaler']
                self.target_scaler = pipeline_data['target_scaler']
                self.metrics = pipeline_data['metrics']
                self.symbol = pipeline_data['symbol']
                self.target_col = pipeline_data['target_col']
                self.prediction_horizon = pipeline_data['prediction_horizon']
                self.window_size = pipeline_data['window_size']
                
                logger.info(f"Pipeline loaded from '{pipeline_path}'")
                
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_future(self, days=5, use_cached_data=True, cache_dir='data'):
        """
        Make predictions for future days based on the latest data.
        
        Parameters:
        -----------
        days : int
            Number of days to predict into the future
        use_cached_data : bool
            Whether to use cached data if available
        cache_dir : str
            Directory to store/retrieve cached data
            
        Returns:
        --------
        pd.DataFrame
            Predictions for future days
        """
        try:
            logger.info(f"Predicting stock prices for the next {days} days...")

            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")

            # Get the latest data
            latest_data = self.fetch_data(use_cached=use_cached_data, cache_dir=cache_dir)
            
            # Engineer features
            latest_features = self.engineer_features()
            
            # Prepare the last window_size days for prediction
            X_latest = latest_features.iloc[-self.window_size:][latest_features.columns.drop(f'target_{self.prediction_horizon}d')].values
            
            # Scale the features
            if self.scaler:
                try:
                    X_latest = self.scaler.transform(X_latest)
                except ValueError as e:
                    # Handle dimension mismatch by making features match the expected dimensions
                    logger.warning(f"Dimension mismatch: {e}")
                    logger.info(f"Feature shapes - Expected: {self.scaler.n_features_in_}, Got: {X_latest.shape[1]}")
                    
                    # If we have more features than expected, trim
                    if X_latest.shape[1] > self.scaler.n_features_in_:
                        logger.info(f"Trimming features from {X_latest.shape[1]} to {self.scaler.n_features_in_}")
                        # Keep only the first n_features_in_ columns
                        X_latest = X_latest[:, :self.scaler.n_features_in_]
                        X_latest = self.scaler.transform(X_latest)
                
            # Reshape for LSTM input
            X_latest = X_latest.reshape(1, self.window_size, X_latest.shape[1])
            
            predictions = []
            current_sequence = X_latest.copy()
            
            for i in range(days):
                # Make prediction
                pred = self.model.predict(current_sequence)
                
                # Inverse transform if scaled
                if hasattr(self, 'target_scaler'):
                    pred_price = self.target_scaler.inverse_transform(pred)[0][0]
                else:
                    pred_price = pred[0][0]
                
                # Create next day's date
                last_date = latest_features.index[-1] + pd.Timedelta(days=i)
                next_date = last_date + pd.Timedelta(days=1)
                
                # Skip weekends
                while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    next_date += pd.Timedelta(days=1)
                
                # Store prediction
                predictions.append({
                    'Date': next_date,
                    'Predicted_Close': pred_price
                })
                
                # Update the sequence for the next prediction
                # This is a simplification - in a real implementation, you would need
                # to update all features, not just the target
                if i < days - 1:
                    # Create a new row for the predicted value
                    # This is simplified - would need actual feature values in practice
                    new_row = np.zeros((1, X_latest.shape[2]))
                    
                    # Shift the sequence one step forward and add the new prediction
                    current_sequence = np.append(current_sequence[:, 1:, :], 
                                               np.expand_dims(new_row, axis=1), 
                                               axis=1)
            
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame(predictions)
            # Convert dates to timezone-aware format for consistency
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'], utc=True)
            predictions_df.set_index('Date', inplace=True)
            
            logger.info("Future predictions complete")
            
            # Return both DataFrame and list format for flexibility
            pred_list = predictions_df['Predicted_Close'].values
            return pred_list if len(pred_list) == days else pred_list[:days]
            
        except Exception as e:
            logger.error(f"Error in future prediction: {str(e)}")
            raise

    def get_future_dates(self, days=5):
        """
        Generate future dates for predictions.
        
        Parameters:
        -----------
        days : int
            Number of days to generate
            
        Returns:
        --------
        list
            List of future dates (as datetime objects) excluding weekends
        """
        try:
            # Start from the latest date in the data or current date if no data
            if self.data is not None and not self.data.empty:
                last_date = self.data.index[-1]
            else:
                last_date = pd.Timestamp.now()
                
            future_dates = []
            current_date = last_date
            
            while len(future_dates) < days:
                # Move to next day
                current_date = current_date + pd.Timedelta(days=1)
                
                # Skip weekends (5 = Saturday, 6 = Sunday)
                if current_date.weekday() < 5:
                    future_dates.append(current_date)
                    
            return future_dates
            
        except Exception as e:
            logger.error(f"Error generating future dates: {str(e)}")
            raise

    def run_pipeline(self, window_size=60, epochs=50, batch_size=32, patience=10,
                     visualize=True, save_model=True, use_cached_data=True, cache_dir='data'):
        """
        Run the complete pipeline.
        
        Parameters:
        -----------
        window_size : int
            Size of the time window for LSTM
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
        visualize : bool
            Whether to generate visualizations
        save_model : bool
            Whether to save the trained model
        use_cached_data : bool
            Whether to use cached data if available
        cache_dir : str
            Directory to store/retrieve cached data
            
        Returns:
        --------
        tuple
            (model, metrics)
        """
        try:
            logger.info(f"Starting LSTM pipeline for {self.symbol}...")
            
            # Step 1: Fetch data
            data = self.fetch_data(use_cached=use_cached_data, cache_dir=cache_dir)
            
            # Step 2: Engineer features
            featured_data = self.engineer_features()
            
            # Step 3: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(window_size=window_size)
            
            # Step 4: Build and train model
            model, history = self.train_model(epochs=epochs, batch_size=batch_size, patience=patience)
            self.history = history
            
            # Step 5: Evaluate model
            metrics = self.evaluate_model()
            
            # Step 6: Visualize results (if requested)
            if visualize:
                # Create visualization directory if needed
                os.makedirs('visualizations', exist_ok=True)
                save_path = os.path.join('visualizations', f'lstm_training_history_{self.symbol}.png')
                self.visualize_results(save_path=save_path)
            
            # Step 7: Save model (if requested)
            if save_model:
                # Create models directory if needed
                os.makedirs('models', exist_ok=True)
                self.save_model(model_path='models/lstm_stock_model.h5',
                              pipeline_path=f'models/{self.symbol}_lstm_pipeline.joblib')
            
            logger.info("LSTM pipeline completed successfully")
            
            return self.model, self.metrics
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = LSTMStockPredictor(
        symbol="AAPL",
        period="5y",
        test_size=0.2,
        target_col='Close',
        prediction_horizon=1
    )
    
    # Run the pipeline
    model, metrics = predictor.run_pipeline(
        window_size=60,
        epochs=50,
        batch_size=32,
        patience=10,
        use_cached_data=True,
        cache_dir='data'
    )
    
    # Make future predictions
    future_predictions = predictor.predict_future(days=5)
    print("\nPredictions for the next 5 trading days:")
    print(future_predictions)
