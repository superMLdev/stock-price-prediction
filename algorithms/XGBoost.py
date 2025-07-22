import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import optuna
import joblib
import os
import logging
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import warnings
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('xgboost_stock_model.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")


class StockPricePredictor:
    """
    End-to-end pipeline for predicting stock prices using XGBoost

    This class handles:
    - Data fetching from Yahoo Finance
    - Feature engineering
    - Data preprocessing
    - Model training with hyperparameter tuning
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
        self.feature_importance = None
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
        self.best_params = None

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
        Create features for XGBoost training.

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
                
            if self.data is None or self.data.empty:
                raise ValueError("No data available for feature engineering")

            df = self.data.copy()

            # Add technical indicators
            df = self.add_technical_indicators(df)

            # Price-based features
            for i in range(1, lookback_days + 1):
                # Lag features
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
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
            for window in [5, 10, 20, 50]:
                df[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window=window).std()

                # Price momentum
                df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1

                # Volume trend
                df[f'volume_trend_{window}d'] = df['Volume'] / df['Volume'].rolling(window=window).mean() - 1

            # Price ratios
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']

            # Day of week, month, quarter features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year

            # Cyclical encoding of time features
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

    def preprocess_data(self, scale_features=True, scaler_type='robust'):
        """
        Preprocess data for model training, including scaling.

        Parameters:
        -----------
        scale_features : bool
            Whether to scale the features
        scaler_type : str
            Type of scaler to use ('standard' or 'robust')

        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Preprocessing data...")

            if self.featured_data is None:
                self.engineer_features()
                
            if self.featured_data is None or self.featured_data.empty:
                raise ValueError("No featured data available for preprocessing")

            # Prepare features and target
            target_col = f'target_{self.prediction_horizon}d'
            feature_columns = [col for col in self.featured_data.columns
                              if col not in [target_col, 'Dividends', 'Stock Splits']]

            self.X = self.featured_data[feature_columns]
            self.y = self.featured_data[target_col]

            # Split data (use time-based split for time series)
            split_idx = int(len(self.featured_data) * (1 - self.test_size))
            split_date = self.featured_data.index[split_idx]

            self.X_train = self.X[self.X.index < split_date]
            self.X_test = self.X[self.X.index >= split_date]
            self.y_train = self.y[self.y.index < split_date]
            self.y_test = self.y[self.y.index >= split_date]

            # Scale features if requested
            if scale_features:
                if scaler_type == 'standard':
                    self.scaler = StandardScaler()
                else:  # robust is better for financial data with outliers
                    self.scaler = RobustScaler()

                # Fit on training data only
                X_train_scaled = self.scaler.fit_transform(self.X_train)
                X_test_scaled = self.scaler.transform(self.X_test)

                # Convert back to dataframes with the original index and column names
                self.X_train = pd.DataFrame(X_train_scaled,
                                           index=self.X_train.index,
                                           columns=self.X_train.columns)
                self.X_test = pd.DataFrame(X_test_scaled,
                                          index=self.X_test.index,
                                          columns=self.X_test.columns)

            logger.info(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def tune_hyperparameters(self, n_trials=100):
        """
        Tune XGBoost hyperparameters with Optuna.

        Parameters:
        -----------
        n_trials : int
            Number of Optuna trials for hyperparameter tuning

        Returns:
        --------
        dict
            Best hyperparameters
        """
        try:
            logger.info(f"Tuning hyperparameters with Optuna ({n_trials} trials)...")

            def objective(trial):
                params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
                    'random_state': 42,
                    'callbacks': [xgb.callback.EarlyStopping(rounds=50)]
                }

                tscv = TimeSeriesSplit(n_splits=5)
                model = xgb.XGBRegressor(**params)

                scores = []
                for train_idx, val_idx in tscv.split(self.X_train):
                    X_train_cv, X_val_cv = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                    y_train_cv, y_val_cv = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                    model.fit(X_train_cv, y_train_cv,
                            eval_set=[(X_val_cv, y_val_cv)],
                            verbose=False)

                    preds = model.predict(X_val_cv)
                    rmse = np.sqrt(mean_squared_error(y_val_cv, preds))
                    scores.append(rmse)

                return np.mean(scores)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)

            logger.info("Best hyperparameters:")
            for key, value in study.best_params.items():
                logger.info(f"    {key}: {value}")

            self.best_params = study.best_params
            return self.best_params

        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise

    def train_model(self, tuned_params=None):
        """
        Train XGBoost model with optimized hyperparameters.

        Parameters:
        -----------
        tuned_params : dict, optional
            Hyperparameters for XGBoost

        Returns:
        --------
        xgboost.XGBRegressor
            Trained model
        """
        try:
            logger.info("Training XGBoost model...")
            
            # Make sure we have data to train on
            if self.X_train is None or self.y_train is None:
                self.preprocess_data()
                
            if self.X_train is None or self.X_train.empty or self.y_train is None or self.y_train.empty:
                raise ValueError("No training data available")

            if tuned_params:
                xgb_params = tuned_params.copy()
            else:
                # Default parameters if no tuning was done
                xgb_params = {
                    'objective': 'reg:squarederror',
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'n_estimators': 1000,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }

            # Remove early_stopping_rounds from best_params if present
            if tuned_params and 'early_stopping_rounds' in tuned_params:
                tuned_params.pop('early_stopping_rounds')

            self.model = xgb.XGBRegressor(**xgb_params)

            # Fit with early stopping (only as direct argument)
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_test, self.y_test)],
                verbose=100
            )

            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"Model trained with {self.model.n_estimators} trees")
            logger.info(f"Top 5 important features: {', '.join(self.feature_importance['feature'].head(5).tolist())}")

            return self.model

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
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
            self.y_pred_train = self.model.predict(self.X_train)
            self.y_pred_test = self.model.predict(self.X_test)

            # Calculate metrics
            metrics = {}

            # Training metrics
            metrics['train_rmse'] = np.sqrt(mean_squared_error(self.y_train, self.y_pred_train))
            metrics['train_mae'] = mean_absolute_error(self.y_train, self.y_pred_train)
            metrics['train_r2'] = r2_score(self.y_train, self.y_pred_train)

            # Testing metrics
            metrics['test_rmse'] = np.sqrt(mean_squared_error(self.y_test, self.y_pred_test))
            metrics['test_mae'] = mean_absolute_error(self.y_test, self.y_pred_test)
            metrics['test_r2'] = r2_score(self.y_test, self.y_pred_test)

            # Calculate MAPE (Mean Absolute Percentage Error)
            metrics['train_mape'] = np.mean(np.abs((self.y_train - self.y_pred_train) / self.y_train)) * 100
            metrics['test_mape'] = np.mean(np.abs((self.y_test - self.y_pred_test) / self.y_test)) * 100

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

            # 1. Actual vs Predicted plot
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(self.y_test.index, self.y_test.values, label='Actual', color='blue', linewidth=2)
            ax1.plot(self.y_test.index, self.y_pred_test, label='Predicted', color='red', linestyle='--', linewidth=2)
            ax1.set_title(f'{self.symbol} - Actual vs Predicted Price', fontsize=14)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Stock Price', fontsize=12)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Add confidence bands if we have validation data
            if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
                # Use the test set to calculate prediction standard deviation
                residuals = self.y_test - self.y_pred_test
                residual_std = np.std(residuals)

                # Plot confidence bands (±2 standard deviations)
                ax1.fill_between(self.y_test.index,
                                self.y_pred_test - 2*residual_std,
                                self.y_pred_test + 2*residual_std,
                                color='red', alpha=0.2, label='95% Confidence Interval')

            # 2. Feature Importance plot
            ax2 = plt.subplot(2, 2, 2)
            top_features = self.feature_importance.head(15)
            top_features = top_features.sort_values('importance')
            bars = ax2.barh(top_features['feature'], top_features['importance'], color=sns.color_palette("viridis", len(top_features)))
            ax2.set_title('Top 15 Feature Importance', fontsize=14)
            ax2.set_xlabel('Importance (gain)', fontsize=12)
            ax2.tick_params(axis='y', labelsize=10)

            # Add values to the bars
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', va='center', fontsize=8)

            # 3. Residuals plot
            ax3 = plt.subplot(2, 2, 3)
            residuals = self.y_test - self.y_pred_test
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
            abs_error = np.abs(self.y_test - self.y_pred_test)
            error_df = pd.DataFrame({'Date': self.y_test.index, 'Absolute Error': abs_error})

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

            # Additional plot for cumulative returns (trading simulation)
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

            # Get actual price changes
            actual_returns = self.y_test.pct_change().fillna(0)

            # Get predicted price changes
            predicted_returns = pd.Series(self.y_pred_test, index=self.y_test.index).pct_change().fillna(0)

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
            plt.plot(cum_strategy_returns, label='Model Strategy', color='green', linewidth=2)

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

            plt.title(f'{self.symbol} - Trading Simulation (Test Period)', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
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

    def backtest(self, test_periods=3):
        """
        Perform walk-forward backtesting of the model.

        Parameters:
        -----------
        test_periods : int
            Number of test periods to evaluate

        Returns:
        --------
        pd.DataFrame
            Backtest results
        """
        try:
            logger.info(f"Starting walk-forward backtesting with {test_periods} periods...")

            if self.featured_data is None:
                self.engineer_features()

            # Prepare data
            target_col = f'target_{self.prediction_horizon}d'
            feature_columns = [col for col in self.featured_data.columns
                              if col not in [target_col, 'Dividends', 'Stock Splits']]

            X = self.featured_data[feature_columns]
            y = self.featured_data[target_col]

            # Create multiple train-test splits
            period_size = len(self.featured_data) // (test_periods + 1)

            results = []

            for i in range(test_periods):
                # Calculate split indices
                test_start_idx = len(self.featured_data) - (i + 1) * period_size
                test_end_idx = len(self.featured_data) - i * period_size if i > 0 else len(self.featured_data)

                # Split data
                test_indices = range(test_start_idx, test_end_idx)
                train_indices = range(0, test_start_idx)

                # Get split data
                X_train = X.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_train = y.iloc[train_indices]
                y_test = y.iloc[test_indices]

                # Train model
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    learning_rate=0.1,
                    max_depth=6,
                    n_estimators=500,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )

                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Store results
                period_results = {
                    'period': i + 1,
                    'start_date': X_test.index[0].strftime('%Y-%m-%d'),
                    'end_date': X_test.index[-1].strftime('%Y-%m-%d'),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }

                results.append(period_results)

                logger.info(f"Period {i+1} ({period_results['start_date']} to {period_results['end_date']}):")
                logger.info(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            results_df = pd.DataFrame(results)

            # Calculate average metrics
            avg_metrics = {
                'avg_rmse': results_df['rmse'].mean(),
                'avg_mae': results_df['mae'].mean(),
                'avg_r2': results_df['r2'].mean()
            }

            logger.info("\nBacktesting Summary:")
            logger.info(f"Average RMSE: {avg_metrics['avg_rmse']:.4f}")
            logger.info(f"Average MAE: {avg_metrics['avg_mae']:.4f}")
            logger.info(f"Average R²: {avg_metrics['avg_r2']:.4f}")

            return results_df

        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            raise

    def save_model(self, filename="xgboost_stock_model.json", save_pipeline=True, model_dir='models'):
        """
        Save the trained model and optionally the full pipeline.

        Parameters:
        -----------
        filename : str
            Name of the file to save the model
        save_pipeline : bool
            Whether to save the full pipeline including scaler
        model_dir : str
            Directory to save model files
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Construct full model path
            model_path = os.path.join(model_dir, filename)
            
            # Save XGBoost model
            self.model.save_model(model_path)
            logger.info(f"Model saved as '{model_path}'")

            if save_pipeline:
                # Save the full pipeline
                pipeline_filename = os.path.join(model_dir, f"{self.symbol}_pipeline.joblib")

                pipeline_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_importance': self.feature_importance,
                    'metrics': self.metrics,
                    'best_params': self.best_params,
                    'symbol': self.symbol,
                    'target_col': self.target_col,
                    'prediction_horizon': self.prediction_horizon,
                    'X_columns': list(self.X.columns)
                }

                joblib.dump(pipeline_data, pipeline_filename)
                logger.info(f"Full pipeline saved as '{pipeline_filename}'")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filename="xgboost_stock_model.json", load_pipeline=False, model_dir='models'):
        """
        Load a trained model and optionally the full pipeline.

        Parameters:
        -----------
        filename : str
            Name of the file to load the model from
        load_pipeline : bool
            Whether to load the full pipeline
        model_dir : str
            Directory where model files are stored

        Returns:
        --------
        xgboost.XGBRegressor or dict
            Loaded model or pipeline
        """
        try:
            # Construct full model path
            model_path = os.path.join(model_dir, filename)
            
            if load_pipeline:
                # Try to find pipeline file for this symbol
                pipeline_filename = os.path.join(model_dir, f"{self.symbol}_pipeline.joblib")
                
                if not os.path.exists(pipeline_filename):
                    # Fall back to default pipeline file
                    pipeline_filename = os.path.join(model_dir, "xgboost_stock_model.joblib")
                
                logger.info(f"Loading pipeline from '{pipeline_filename}'")
                pipeline_data = joblib.load(pipeline_filename)

                self.model = pipeline_data['model']
                self.scaler = pipeline_data['scaler']
                self.feature_importance = pipeline_data['feature_importance']
                self.metrics = pipeline_data['metrics']
                self.best_params = pipeline_data['best_params']
                self.symbol = pipeline_data['symbol']
                self.target_col = pipeline_data['target_col']
                self.prediction_horizon = pipeline_data['prediction_horizon']

                logger.info(f"Full pipeline loaded from '{pipeline_filename}'")
                return pipeline_data
            else:
                logger.info(f"Loading model from '{model_path}'")
                self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
                logger.info(f"Model loaded from '{model_path}'")
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

            # Engineer features for the latest data
            latest_features = self.engineer_features()

            # Store column names if self.X is None (happens when model was loaded without training)
            if self.X is None and hasattr(self, 'model') and self.model is not None:
                # Get feature names from the model
                feature_names = self.model.get_booster().feature_names
                if feature_names:
                    logger.info(f"Using {len(feature_names)} feature names from loaded model")
                else:
                    # Use all features from latest_features as a fallback
                    feature_names = latest_features.columns.tolist()
                    # Remove target column if present
                    target_col = f'target_{self.prediction_horizon}d'
                    if target_col in feature_names:
                        feature_names.remove(target_col)
                    logger.info(f"Using {len(feature_names)} features from current data")
            else:
                # Normal case - we have self.X from training
                feature_names = self.X.columns.tolist()

            # Check if all feature_names are in latest_features
            missing_features = [f for f in feature_names if f not in latest_features.columns]
            extra_features = [f for f in latest_features.columns if f not in feature_names and f != f'target_{self.prediction_horizon}d']
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features from the model: {missing_features[:5]}...")
                # Add missing features with zeros
                for feat in missing_features:
                    latest_features[feat] = 0
                    
            if extra_features:
                logger.warning(f"Found {len(extra_features)} extra features not used by the model: {extra_features[:5]}...")
                # We'll ignore these extra features
            
            predictions = []
            current_data = latest_features.iloc[-1:].copy()

            # Prepare the target column name to ignore
            target_col = f'target_{self.prediction_horizon}d'
            if target_col in current_data.columns:
                current_data = current_data.drop(columns=[target_col])

            for i in range(days):
                # Prepare features for prediction - ensure only using columns the model knows about
                pred_features = current_data[feature_names].copy()

                # Scale features if necessary
                if self.scaler:
                    try:
                        pred_features_scaled = self.scaler.transform(pred_features)
                        pred_features = pd.DataFrame(pred_features_scaled,
                                                   index=pred_features.index,
                                                   columns=pred_features.columns)
                    except ValueError as e:
                        logger.warning(f"Scaler error: {e}. Attempting to fix...")
                        # If there's a dimension mismatch, try to adapt
                        if pred_features.shape[1] != self.scaler.n_features_in_:
                            if pred_features.shape[1] > self.scaler.n_features_in_:
                                # If we have more features than expected, keep only what's needed
                                logger.info(f"Trimming features from {pred_features.shape[1]} to {self.scaler.n_features_in_}")
                                # Assume the first n features match what the scaler expects
                                pred_features = pred_features.iloc[:, :self.scaler.n_features_in_]
                                pred_features_scaled = self.scaler.transform(pred_features)
                                pred_features = pd.DataFrame(pred_features_scaled,
                                                          index=pred_features.index,
                                                          columns=pred_features.columns)
                            else:
                                # If we have fewer features, this is more complex
                                # Skip scaling as a last resort
                                logger.warning("Not enough features for scaling - using unscaled features")
                                pass

                # Make prediction
                try:
                    pred_price = self.model.predict(pred_features)[0]
                except Exception as pred_error:
                    logger.error(f"Prediction error: {pred_error}")
                    # Fallback to returning the last known price
                    if 'Close' in latest_features.columns:
                        pred_price = latest_features['Close'].iloc[-1]
                        logger.info(f"Using last known price {pred_price} as fallback")
                    else:
                        # If we can't even find the last price, return 0
                        pred_price = 0
                        logger.warning("Using 0 as fallback prediction")

                # Create next day's date
                last_date = current_data.index[-1]
                next_date = last_date + pd.Timedelta(days=1)

                # Skip weekends
                while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    next_date += pd.Timedelta(days=1)

                # Store prediction
                predictions.append({
                    'Date': next_date,
                    'Predicted_Close': pred_price
                })

                # Update features for next prediction (a simple approach)
                # In a real application, this would need more sophisticated updates to all features
                new_row = current_data.copy()
                new_row.index = [next_date]

                # Update basic features for next day
                if 'Close' in new_row.columns:
                    new_row['Close'] = pred_price

                # Update lag features
                for col in pred_features.columns:
                    if 'lag_1' in col:
                        base_col = col.replace('_lag_1', '')
                        if base_col in new_row.columns:
                            new_row[col] = new_row[base_col]
                    elif 'lag_' in col:
                        parts = col.split('_lag_')
                        if len(parts) == 2:
                            base_col = parts[0]
                            lag = int(parts[1])
                            prev_lag = lag - 1
                            prev_lag_col = f"{base_col}_lag_{prev_lag}"
                            if prev_lag_col in pred_features.columns:
                                new_row[col] = pred_features[prev_lag_col].iloc[0]lag_num = int(col.split('_lag_')[1])
                        base_col = col.replace(f'_lag_{lag_num}', '')
                        lag_col_prev = f"{base_col}_lag_{lag_num-1}"
                        if lag_col_prev in new_row.columns:
                            new_row[col] = new_row[lag_col_prev]

                # Update current data for next iteration
                current_data = new_row.copy()

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
            # Return a reasonable fallback prediction in case of error
            # For example, predict the last known price for all future days
            if hasattr(self, 'featured_data') and self.featured_data is not None and not self.featured_data.empty:
                last_price = self.featured_data['Close'].iloc[-1]
                logger.info(f"Using last known price {last_price} as fallback prediction")
                return [last_price] * days
            else:
                logger.error("No data available for fallback prediction")
                return [0] * days
            
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

    def run_pipeline(self, tune=True, n_trials=50, visualize=True, save_model=True, backtest=False, scale_features=True, use_cached_data=True, cache_dir='data'):
        """
        Run the complete pipeline.

        Parameters:
        -----------
        tune : bool
            Whether to tune hyperparameters
        n_trials : int
            Number of Optuna trials for hyperparameter tuning
        visualize : bool
            Whether to generate visualizations
        save_model : bool
            Whether to save the trained model
        backtest : bool
            Whether to perform backtesting
        scale_features : bool
            Whether to scale features
        use_cached_data : bool
            Whether to use cached data if available
        cache_dir : str
            Directory to store/retrieve cached data

        Returns:
        --------
        tuple
            (model, metrics, feature_importance)
        """
        try:
            logger.info(f"Starting prediction pipeline for {self.symbol}...")

            # Step 1: Fetch data
            self.fetch_data(use_cached=use_cached_data, cache_dir=cache_dir)

            # Step 2: Engineer features
            self.engineer_features()

            # Step 3: Preprocess data
            self.preprocess_data(scale_features=scale_features)

            # Step 4: Tune hyperparameters (if requested)
            if tune:
                self.tune_hyperparameters(n_trials=n_trials)
                self.train_model(self.best_params)
            else:
                self.train_model()

            # Step 5: Evaluate model
            self.evaluate_model()

            # Step 6: Generate visualizations (if requested)
            if visualize:
                self.visualize_results()

            # Step 7: Perform backtesting (if requested)
            if backtest:
                self.backtest()

            # Step 8: Save model (if requested)
            if save_model:
                self.save_model(save_pipeline=True)

            logger.info("Pipeline completed successfully")
            return self.model, self.metrics, self.feature_importance

        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = StockPricePredictor(
        symbol="AAPL",             # Stock ticker symbol
        period="5y",               # Data period to fetch
        test_size=0.2,             # Proportion of data for testing
        target_col='Close',        # Target column to predict
        prediction_horizon=1       # Days ahead to predict
    )

    # Run the complete pipeline
    model, metrics, importance = predictor.run_pipeline(
        tune=True,                 # Use Optuna for hyperparameter tuning
        n_trials=50,               # Number of Optuna trials
        visualize=True,            # Generate visualizations
        save_model=True,           # Save the model
        backtest=True,             # Perform backtesting
        scale_features=True,       # Scale the features
        use_cached_data=True,      # Use cached data if available
        cache_dir='data'           # Directory to store/retrieve cached data
    )

    # Make predictions for the next 5 days
    future_predictions = predictor.predict_future(
        days=5,
        use_cached_data=True,
        cache_dir='data'
    )
    print("\nPredictions for the next 5 trading days:")
    print(future_predictions)
