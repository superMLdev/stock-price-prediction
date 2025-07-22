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
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transformer_prediction.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
except:
    logger.warning("Unable to set memory growth for GPU")


class TransformerBlock(layers.Layer):
    """Transformer block implementation for time series."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Self-attention with residual connection and layer normalization
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """Positional encoding layer to give the model information about the position of each element."""

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # Apply sin to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class StockPriceTransformer:
    """
    End-to-end pipeline for predicting stock prices using a Transformer model.

    This class handles:
    - Data fetching from Yahoo Finance
    - Feature engineering
    - Data preprocessing
    - Model building and training
    - Evaluation and visualization
    - Model persistence
    """

    def __init__(self, symbol, period="5y", test_size=0.2, target_col='Close', prediction_horizon=1,
                 sequence_length=30, batch_size=32):
        """
        Initialize the stock price prediction pipeline with Transformer model.

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
        sequence_length : int
            Number of time steps to use for prediction
        batch_size : int
            Batch size for training
        """
        self.symbol = symbol
        self.period = period
        self.test_size = test_size
        self.target_col = target_col
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # Initialize attributes
        self.model = None
        self.data = None
        self.featured_data = None
        self.scaler = None
        self.target_scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.metrics = None
        self.history = None

    def fetch_data(self, use_cached=False, cache_dir='data'):
        """
        Fetch historical stock data from Yahoo Finance or from cache if available.

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
            # Create cache path
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{self.symbol}_{self.period}_data.csv")
            
            # Check if cached data exists and is requested
            if os.path.exists(cache_path) and use_cached:
                logging.info(f"Loading cached data for {self.symbol} from {cache_path}")
                self.data = pd.read_csv(cache_path)
                # Properly convert the index to datetime with timezone handling
                self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
                self.data.set_index('Date', inplace=True)
                # Remove timezone information for consistency with yfinance data
                self.data.index = self.data.index.tz_localize(None)
                logging.info(f"Loaded {len(self.data)} rows of data from cache")
            else:
                # Fetch data from Yahoo Finance
                logging.info(f"Fetching data for {self.symbol} from Yahoo Finance...")
                stock = yf.Ticker(self.symbol)
                self.data = stock.history(period=self.period)

                # Check if we got any data
                if self.data.empty:
                    raise ValueError(f"No data returned for symbol {self.symbol}")

                # Handle missing values
                if self.data.isnull().sum().sum() > 0:
                    logging.warning(f"Found {self.data.isnull().sum().sum()} missing values in raw data")
                    self.data = self.data.fillna(method='ffill')
                
                # Save to cache if directory exists
                if use_cached:
                    # Reset index to include Date as a column, then save
                    self.data.reset_index().to_csv(cache_path, index=False)
                    logging.info(f"Saving {len(self.data)} rows of data to {cache_path}")

                logging.info(f"Downloaded {len(self.data)} rows of data")
            
            return self.data

        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
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

        return df

    def engineer_features(self, lookback_days=30):
        """
        Create features for model training.

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
                # Lag features - only for the target column for transformer
                df[f'{self.target_col.lower()}_lag_{i}'] = df[self.target_col].shift(i)

                # Return-based features
                if i <= 10:  # Only create return features for shorter periods
                    df[f'return_{i}d'] = df['Close'].pct_change(i)
                    df[f'volume_change_{i}d'] = df['Volume'].pct_change(i)

            # Price gaps
            df['gap_open'] = df['Open'] / df['Close'].shift(1) - 1
            df['gap_close'] = df['Close'] / df['Open'] - 1

            # Volatility measures
            for window in [5, 10, 20, 50]:
                df[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window=window).std()

                # Price momentum
                df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1

            # Day of week, month, quarter features (for time-based patterns)
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter

            # Cyclical encoding of time features (better for ML models)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

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

    def create_sequences(self, X, y, sequence_length):
        """
        Create sequences for time series prediction.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature data
        y : numpy.ndarray
            Target data
        sequence_length : int
            Length of each sequence

        Returns:
        --------
        tuple
            (X_seq, y_seq) where X_seq contains sequences of features and y_seq contains target values
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def preprocess_data(self):
        """
        Preprocess data for model training, including scaling and sequence creation.

        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test as sequences
        """
        try:
            logger.info("Preprocessing data...")

            if self.featured_data is None:
                self.engineer_features()

            # Prepare features and target
            target_col = f'target_{self.prediction_horizon}d'
            feature_columns = [col for col in self.featured_data.columns
                              if col not in [target_col, 'Dividends', 'Stock Splits']]

            # Split data (use time-based split for time series)
            split_idx = int(len(self.featured_data) * (1 - self.test_size))

            # Get features and target
            X = self.featured_data[feature_columns].values
            y = self.featured_data[target_col].values.reshape(-1, 1)

            # Scale features
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Separate scaler for the target
            self.target_scaler = MinMaxScaler()
            y_scaled = self.target_scaler.fit_transform(y)

            # Split into train and test sets
            X_train_scaled = X_scaled[:split_idx]
            X_test_scaled = X_scaled[split_idx:]
            y_train_scaled = y_scaled[:split_idx]
            y_test_scaled = y_scaled[split_idx:]

            # Create sequences
            self.X_train, self.y_train = self.create_sequences(
                X_train_scaled, y_train_scaled, self.sequence_length
            )
            self.X_test, self.y_test = self.create_sequences(
                X_test_scaled, y_test_scaled, self.sequence_length
            )

            # Save original y values for later evaluation
            self.y_train_orig = y[:split_idx][self.sequence_length:]
            self.y_test_orig = y[split_idx:][self.sequence_length:]

            logger.info(f"Training sequences: {len(self.X_train)}, Test sequences: {len(self.X_test)}")
            logger.info(f"Input shape: {self.X_train.shape}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def build_model(self, num_features, head_size=256, num_heads=4, ff_dim=256, num_transformer_blocks=4,
                   mlp_units=[128, 64], dropout=0.1, mlp_dropout=0.2):
        """
        Build a Transformer model for time series forecasting.

        Parameters:
        -----------
        num_features : int
            Number of features in the input data
        head_size : int
            Size of each attention head
        num_heads : int
            Number of attention heads
        ff_dim : int
            Hidden layer size in feed forward network inside transformer
        num_transformer_blocks : int
            Number of transformer blocks
        mlp_units : list
            Hidden layer sizes for the final MLP
        dropout : float
            Dropout rate for transformer
        mlp_dropout : float
            Dropout rate for the final MLP

        Returns:
        --------
        tf.keras.Model
            Compiled Transformer model
        """
        try:
            logger.info("Building Transformer model...")

            # Input shape is (batch_size, sequence_length, num_features)
            inputs = keras.Input(shape=(self.sequence_length, num_features))

            # Add positional encoding
            x = PositionalEncoding(self.sequence_length, num_features)(inputs)

            # Add transformer blocks
            for _ in range(num_transformer_blocks):
                x = TransformerBlock(num_features, num_heads, ff_dim, dropout)(x)

            # Use attention pooling to get a fixed-size output
            x = layers.GlobalAveragePooling1D()(x)

            # Add MLP layers
            for dim in mlp_units:
                x = layers.Dense(dim, activation="relu")(x)
                x = layers.Dropout(mlp_dropout)(x)

            # Final output layer (single value for regression)
            outputs = layers.Dense(1)(x)

            # Build and compile model
            model = keras.Model(inputs=inputs, outputs=outputs)

            # Compile with Adam optimizer and MSE loss
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss="mse",
                metrics=["mae", "mse", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
            )

            model.summary(print_fn=logger.info)

            self.model = model
            return model

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train_model(self, epochs=50, patience=10, verbose=1):
        """
        Train the Transformer model.

        Parameters:
        -----------
        epochs : int
            Maximum number of epochs to train
        patience : int
            Number of epochs with no improvement after which training will be stopped
        verbose : int
            Verbosity mode

        Returns:
        --------
        tf.keras.Model
            Trained model
        """
        try:
            logger.info("Training Transformer model...")

            if self.model is None:
                self.build_model(num_features=self.X_train.shape[2])

            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                restore_best_weights=True
            )

            # Learning rate scheduler
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )

            # Train the model
            self.history = self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=verbose
            )

            logger.info(f"Model trained for {len(self.history.history['loss'])} epochs")

            return self.model

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

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
            y_pred_train_scaled = self.model.predict(self.X_train)
            y_pred_test_scaled = self.model.predict(self.X_test)

            # Inverse transform to get actual values
            y_pred_train = self.target_scaler.inverse_transform(y_pred_train_scaled)
            y_pred_test = self.target_scaler.inverse_transform(y_pred_test_scaled)
            self.y_pred_train = y_pred_train.flatten()
            self.y_pred_test = y_pred_test.flatten()

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

    def visualize_results(self, save_path=None):
        """
        Create visualizations for model results.

        Parameters:
        -----------
        save_path : str, optional
            Path to save visualization figures
        """
        try:
            logger.info("Creating visualizations...")

            # Create a figure with subplots
            fig = plt.figure(figsize=(20, 15))

            # 1. Training history
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Training and Validation Loss', fontsize=14)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss (MSE)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Get dates for test set
            test_dates = self.featured_data.index[-(len(self.y_pred_test)):]

            # 2. Actual vs Predicted plot
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(test_dates, self.y_test_orig, label='Actual', color='blue', linewidth=2)
            ax2.plot(test_dates, self.y_pred_test, label='Predicted', color='red', linestyle='--', linewidth=2)
            ax2.set_title(f'{self.symbol} - Actual vs Predicted Price', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Stock Price', fontsize=12)
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Calculate prediction standard deviation for confidence bands
            # Ensure shapes match for subtraction
            logger.info(f"Before subtraction - y_test_orig shape: {self.y_test_orig.shape if hasattr(self.y_test_orig, 'shape') else len(self.y_test_orig)}, y_pred_test shape: {self.y_pred_test.shape if hasattr(self.y_pred_test, 'shape') else len(self.y_pred_test)}")
            min_len_orig = min(len(self.y_test_orig), len(self.y_pred_test))
            # Explicit conversion to numpy arrays and reshaping to ensure 1D arrays
            y_test_trimmed = np.array(self.y_test_orig[:min_len_orig]).flatten()
            y_pred_trimmed = np.array(self.y_pred_test[:min_len_orig]).flatten()
            logger.info(f"After conversion - y_test_trimmed shape: {y_test_trimmed.shape}, y_pred_trimmed shape: {y_pred_trimmed.shape}")
            
            # Calculate residuals as a 1D array
            residuals = y_test_trimmed - y_pred_trimmed
            logger.info(f"Residuals shape after calculation: {residuals.shape}")
            residual_std = np.std(residuals)

            # Plot confidence bands (±2 standard deviations)
            ax2.fill_between(test_dates[:min_len_orig],
                            y_pred_trimmed - 2*residual_std,
                            y_pred_trimmed + 2*residual_std,
                            color='red', alpha=0.2, label='95% Confidence Interval')

            # 3. Residuals plot
            ax3 = plt.subplot(2, 2, 3)
            # Debug print statements for shapes
            logger.info(f"y_pred_trimmed shape: {y_pred_trimmed.shape}, residuals shape: {residuals.shape}")
            
            # Use the pre-calculated residuals with matching shapes
            ax3.scatter(y_pred_trimmed, residuals, alpha=0.5, color='blue')
            ax3.axhline(y=0, color='r', linestyle='-')

            # Add trend line to residuals (using only the valid data points)
            z = np.polyfit(y_pred_trimmed, residuals, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(y_pred_trimmed), p(sorted(y_pred_trimmed)), "r--", alpha=0.8)

            ax3.set_title('Residuals Plot', fontsize=14)
            ax3.set_xlabel('Predicted Values', fontsize=12)
            ax3.set_ylabel('Residuals', fontsize=12)
            ax3.grid(True, alpha=0.3)

            # 4. Error over time
            ax4 = plt.subplot(2, 2, 4)
            abs_error = np.abs(residuals)  # Use pre-calculated residuals
            logger.info(f"test_dates shape: {len(test_dates[:min_len_orig])}, abs_error shape: {len(abs_error)}")
            
            # Use the matched data
            ax4.plot(test_dates[:min_len_orig], abs_error, color='blue', alpha=0.7)

            # Add trend line for error
            z = np.polyfit(range(len(abs_error)), abs_error, 1)
            p = np.poly1d(z)
            ax4.plot(test_dates[:min_len_orig], p(range(len(abs_error))), "r--", linewidth=2)

            ax4.set_title('Prediction Error Over Time', fontsize=14)
            ax4.set_ylabel('Absolute Error', fontsize=12)
            ax4.set_xlabel('Date', fontsize=12)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                # Check if the save_path has a directory component
                dir_path = os.path.dirname(save_path)
                if dir_path:  # Only try to create directory if there's a directory part
                    os.makedirs(dir_path, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualizations saved to {save_path}")

            plt.show()

            # Also create the trading simulation visualization
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

            # Get test dates
            test_dates = self.featured_data.index[-(len(self.y_pred_test)):]
            
            # Ensure all arrays have the same length
            min_len = min(len(test_dates), len(self.y_test_orig), len(self.y_pred_test))
            logger.info(f"In trading simulation - test_dates: {len(test_dates)}, y_test_orig: {self.y_test_orig.shape}, y_pred_test: {self.y_pred_test.shape}")
            
            # Create dataframe with predictions and actual values
            df_pred = pd.DataFrame({
                'actual': self.y_test_orig.flatten()[:min_len],
                'predicted': self.y_pred_test.flatten()[:min_len]
            }, index=test_dates[:min_len])

            # Calculate returns
            df_pred['actual_return'] = df_pred['actual'].pct_change().fillna(0)
            df_pred['pred_return'] = df_pred['predicted'].pct_change().fillna(0)

            # Simple trading strategy: Buy when predicted return is positive, sell when negative
            df_pred['signal'] = np.sign(df_pred['pred_return'])

            # Calculate strategy returns (shifted by 1 to implement next day)
            df_pred['strategy_return'] = df_pred['signal'].shift(1) * df_pred['actual_return']
            df_pred['strategy_return'] = df_pred['strategy_return'].fillna(0)

            # Calculate cumulative returns
            df_pred['cum_actual_return'] = (1 + df_pred['actual_return']).cumprod() - 1
            df_pred['cum_strategy_return'] = (1 + df_pred['strategy_return']).cumprod() - 1

            # Plot returns
            plt.plot(df_pred.index, df_pred['cum_actual_return'], label='Buy & Hold', color='blue', linewidth=2)
            plt.plot(df_pred.index, df_pred['cum_strategy_return'], label='Transformer Strategy', color='green', linewidth=2)

            # Add annotations for final returns
            final_bh_return = df_pred['cum_actual_return'].iloc[-1]
            final_strategy_return = df_pred['cum_strategy_return'].iloc[-1]

            plt.scatter(df_pred.index[-1], final_bh_return, color='blue', zorder=5)
            plt.scatter(df_pred.index[-1], final_strategy_return, color='green', zorder=5)

            plt.annotate(f'{final_bh_return:.2%}',
                        (df_pred.index[-1], final_bh_return),
                        xytext=(10, 10), textcoords='offset points', fontsize=12)

            plt.annotate(f'{final_strategy_return:.2%}',
                        (df_pred.index[-1], final_strategy_return),
                        xytext=(10, -15), textcoords='offset points', fontsize=12)

            plt.title(f'{self.symbol} - Trading Simulation (Transformer Model)', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('transformer_trading_simulation.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Calculate trading statistics
            n_trades = np.sum(np.abs(df_pred['signal'].diff().fillna(0)) > 0)
            win_rate = np.mean(df_pred['strategy_return'] > 0)
            sharpe = np.sqrt(252) * df_pred['strategy_return'].mean() / df_pred['strategy_return'].std() if df_pred['strategy_return'].std() > 0 else 0

            logger.info(f"\nTrading Simulation Results:")
            logger.info(f"Buy & Hold Return: {final_bh_return:.2%}")
            logger.info(f"Strategy Return: {final_strategy_return:.2%}")
            logger.info(f"Number of Trades: {n_trades}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe:.2f}")

        except Exception as e:
            logger.error(f"Error in trading simulation plot: {str(e)}")

    def predict_future(self, days=5):
        """
        Make predictions for future days based on the latest data.

        Parameters:
        -----------
        days : int
            Number of days to predict into the future

        Returns:
        --------
        pd.DataFrame
            Predictions for future days
        """
        try:
            logger.info(f"Predicting stock prices for the next {days} days...")

            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")

            # Get the latest data for predictions
            latest_data = self.featured_data.copy()

            # Get the latest sequence for prediction
            latest_features = latest_data.drop(columns=[f'target_{self.prediction_horizon}d'])
            latest_scaled = self.scaler.transform(latest_features.values)

            # Take the last sequence_length records
            latest_sequence = latest_scaled[-self.sequence_length:]
            latest_sequence = latest_sequence.reshape(1, self.sequence_length, latest_scaled.shape[1])

            # Make the first prediction
            next_pred_scaled = self.model.predict(latest_sequence)
            next_pred = self.target_scaler.inverse_transform(next_pred_scaled)[0][0]

            # Initialize with first prediction
            predictions = [next_pred]
            prediction_dates = [latest_data.index[-1] + pd.Timedelta(days=1)]

            # Skip weekends for the first date
            while prediction_dates[0].weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                prediction_dates[0] += pd.Timedelta(days=1)

            # Now iteratively predict the rest of the days
            for i in range(1, days):
                # Prepare sequence for next prediction (rolling window approach)
                # Shift the sequence by dropping the first element and appending the new prediction
                # For simplicity, we copy the last row and update just the target column
                next_features = latest_scaled[-1:].copy()  # Copy the last data point
                next_features[0, latest_features.columns.get_loc('Close')] = next_pred_scaled[0][0]  # Update the Close price

                # Update the sequence by removing first element and adding new prediction
                latest_sequence = np.vstack((latest_sequence[0, 1:], next_features))
                latest_sequence = latest_sequence.reshape(1, self.sequence_length, latest_scaled.shape[1])

                # Make prediction
                next_pred_scaled = self.model.predict(latest_sequence)
                next_pred = self.target_scaler.inverse_transform(next_pred_scaled)[0][0]

                # Add to predictions
                predictions.append(next_pred)

                # Calculate next date (skip weekends)
                next_date = prediction_dates[-1] + pd.Timedelta(days=1)
                while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    next_date += pd.Timedelta(days=1)
                prediction_dates.append(next_date)

            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'Predicted_Close': predictions
            }, index=prediction_dates)

            logger.info("Future predictions complete")
            return predictions_df

        except Exception as e:
            logger.error(f"Error in future prediction: {str(e)}")
            raise

    def save_model(self, path="transformer_model"):
        """
        Save the trained model and preprocessing components.

        Parameters:
        -----------
        path : str
            Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            # Save the model
            self.model.save(f"{path}/model.keras")
            logger.info(f"Model saved to {path}/model.keras")

            # Save the scalers
            joblib.dump(self.scaler, f"{path}/feature_scaler.joblib")
            joblib.dump(self.target_scaler, f"{path}/target_scaler.joblib")

            # Save model metadata
            metadata = {
                'symbol': self.symbol,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'target_col': self.target_col,
                'metrics': self.metrics,
                'feature_columns': list(self.featured_data.drop(columns=[f'target_{self.prediction_horizon}d']).columns)
            }

            joblib.dump(metadata, f"{path}/metadata.joblib")
            logger.info(f"Model metadata and preprocessors saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path="transformer_model"):
        """
        Load a trained model and preprocessing components.

        Parameters:
        -----------
        path : str
            Path to load the model from
        """
        try:
            # Load the model
            self.model = keras.models.load_model(f"{path}/model.keras")
            logger.info(f"Model loaded from {path}/model.keras")

            # Load the scalers
            self.scaler = joblib.load(f"{path}/feature_scaler.joblib")
            self.target_scaler = joblib.load(f"{path}/target_scaler.joblib")

            # Load model metadata
            metadata = joblib.load(f"{path}/metadata.joblib")
            self.symbol = metadata['symbol']
            self.sequence_length = metadata['sequence_length']
            self.prediction_horizon = metadata['prediction_horizon']
            self.target_col = metadata['target_col']
            self.metrics = metadata['metrics']

            logger.info(f"Model metadata and preprocessors loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def run_pipeline(self, train=True, epochs=50, visualize=True, save_model=True):
        """
        Run the complete pipeline.

        Parameters:
        -----------
        train : bool
            Whether to train a new model
        epochs : int
            Number of epochs to train
        visualize : bool
            Whether to generate visualizations
        save_model : bool
            Whether to save the trained model

        Returns:
        --------
        tuple
            (model, metrics)
        """
        try:
            logger.info(f"Starting Transformer prediction pipeline for {self.symbol}...")

            # Step 1: Fetch data
            self.fetch_data()

            # Step 2: Engineer features
            self.engineer_features()

            # Step 3: Preprocess data
            self.preprocess_data()

            # Step 4: Build and train model
            if train:
                self.build_model(num_features=self.X_train.shape[2])
                self.train_model(epochs=epochs)

            # Step 5: Evaluate model
            self.evaluate_model()

            # Step 6: Generate visualizations
            if visualize:
                self.visualize_results(save_path='transformer_performance.png')

            # Step 7: Save model
            if save_model:
                self.save_model()

            # Step 8: Predict future prices
            future_predictions = self.predict_future(days=5)
            print("\nTransformer Predictions for the next 5 trading days:")
            print(future_predictions)

            logger.info("Pipeline completed successfully")
            return self.model, self.metrics

        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = StockPriceTransformer(
        symbol="AAPL",             # Stock ticker symbol
        period="5y",               # Data period to fetch
        test_size=0.2,             # Proportion of data for testing
        target_col='Close',        # Target column to predict
        prediction_horizon=1,      # Days ahead to predict
        sequence_length=60,        # Length of sequence for transformer
        batch_size=32              # Batch size for training
    )

    # Run the complete pipeline
    model, metrics = predictor.run_pipeline(
        train=True,                # Train a new model
        epochs=15,                # Maximum number of epochs
        visualize=True,            # Generate visualizations
        save_model=True            # Save the model
    )
