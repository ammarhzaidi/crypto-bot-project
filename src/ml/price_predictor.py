"""
Machine Learning module for cryptocurrency price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging

# Setup logging
logger = logging.getLogger('ml_price_predictor')


class FeatureGenerator:
    """Generates features for ML models from price data."""

    def __init__(self):
        """Initialize the feature generator."""
        self.scalers = {}

    def create_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create features from price data.

        Args:
            df: DataFrame with OHLCV data
            window_sizes: List of window sizes for rolling features

        Returns:
            DataFrame with features
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Basic price features
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))

        # Price difference features
        result['price_diff'] = result['close'] - result['open']
        result['high_low_diff'] = result['high'] - result['low']
        result['close_open_ratio'] = result['close'] / result['open']

        # Volume features
        result['volume_change'] = result['volume'].pct_change()
        result['volume_ma5'] = result['volume'].rolling(window=5).mean()
        result['volume_ma5_ratio'] = result['volume'] / result['volume_ma5']

        # Create features for each window size
        for window in window_sizes:
            # Price momentum
            result[f'close_ma{window}'] = result['close'].rolling(window=window).mean()
            result[f'close_ma{window}_ratio'] = result['close'] / result[f'close_ma{window}']

            # Volatility
            result[f'volatility_{window}'] = result['returns'].rolling(window=window).std()

            # Price direction
            result[f'price_up_{window}'] = (result['close'] > result['close'].shift(window)).astype(int)

            # Min/Max features
            result[f'min_{window}'] = result['low'].rolling(window=window).min()
            result[f'max_{window}'] = result['high'].rolling(window=window).max()
            result[f'close_min_ratio_{window}'] = result['close'] / result[f'min_{window}']
            result[f'close_max_ratio_{window}'] = result['close'] / result[f'max_{window}']

        # Drop NaN values created by rolling windows
        result = result.dropna()

        return result

    def prepare_data_for_training(self,
                                  features_df: pd.DataFrame,
                                  target_column: str = 'close',
                                  prediction_horizon: int = 1,
                                  test_size: float = 0.2,
                                  scale_features: bool = True) -> Tuple:
        """
        Prepare data for training ML models.

        Args:
            features_df: DataFrame with features
            target_column: Column to predict
            prediction_horizon: Number of periods ahead to predict
            test_size: Proportion of data to use for testing
            scale_features: Whether to scale features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Create the target variable (future price)
        df = features_df.copy()
        df[f'future_{target_column}'] = df[target_column].shift(-prediction_horizon)

        # Drop rows with NaN in the target
        df = df.dropna()

        # Separate features and target
        y = df[f'future_{target_column}'].values

        # Drop columns that shouldn't be features
        drop_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', f'future_{target_column}']
        X = df.drop(columns=[col for col in drop_columns if col in df.columns])

        # Store feature names
        feature_names = X.columns.tolist()

        # Convert to numpy array
        X = X.values

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        # Scale features if requested
        if scale_features:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Store the scaler for later use
            self.scalers['features'] = scaler

            # Also create a scaler for the target (for inverse transformation later)
            target_scaler = MinMaxScaler()
            y_train_reshaped = y_train.reshape(-1, 1)
            target_scaler.fit(y_train_reshaped)
            self.scalers['target'] = target_scaler

        return X_train, X_test, y_train, y_test, feature_names


class PricePredictor:
    """ML-based price predictor for cryptocurrencies."""

    def __init__(self, model_dir: str = "data/ml_models"):
        """
        Initialize the price predictor.

        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.feature_generator = FeatureGenerator()
        self.current_symbol = None
        self.current_timeframe = None

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def _get_model_path(self, symbol: str, timeframe: str, model_type: str) -> str:
        """Get the path for saving/loading a model."""
        return os.path.join(self.model_dir, f"{symbol}_{timeframe}_{model_type}.joblib")

    def train_model(self,
                    df: pd.DataFrame,
                    symbol: str,
                    timeframe: str,
                    model_type: str = 'random_forest',
                    prediction_horizon: int = 1,
                    test_size: float = 0.2,
                    hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train a machine learning model for price prediction.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being predicted
            timeframe: Timeframe of the data
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'linear', 'svr')
            prediction_horizon: Number of periods ahead to predict
            test_size: Proportion of data to use for testing
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training results
        """
        # Store current symbol and timeframe
        self.current_symbol = symbol
        self.current_timeframe = timeframe

        # Generate features
        features_df = self.feature_generator.create_features(df)

        # Prepare data for training
        X_train, X_test, y_train, y_test, feature_names = self.feature_generator.prepare_data_for_training(
            features_df,
            prediction_horizon=prediction_horizon,
            test_size=test_size
        )

        # Select model based on type
        if model_type == 'random_forest':
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

        elif model_type == 'gradient_boosting':
            if hyperparameter_tuning:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3)
            else:
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        elif model_type == 'linear':
            model = LinearRegression()

        elif model_type == 'svr':
            if hyperparameter_tuning:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear']
                }
                model = GridSearchCV(SVR(), param_grid, cv=3)
            else:
                model = SVR(kernel='rbf')

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train the model
        model.fit(X_train, y_train)

        # If we did hyperparameter tuning, get the best model
        if hyperparameter_tuning and hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
            logger.info(f"Best parameters: {model.get_params()}")

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store the model
        self.models[model_type] = {
            'model': model,
            'feature_names': feature_names,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'prediction_horizon': prediction_horizon
        }

        # Save the model to disk
        model_path = self._get_model_path(symbol, timeframe, model_type)
        joblib.dump({
            'model': model,
            'feature_names': feature_names,
            'scalers': self.feature_generator.scalers,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'prediction_horizon': prediction_horizon
        }, model_path)

        logger.info(f"Model saved to {model_path}")

        # Return training results
        return {
            'model_type': model_type,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'feature_importance': self._get_feature_importance(model, feature_names) if hasattr(model,
                                                                                                'feature_importances_') else None,
            'prediction_horizon': prediction_horizon
        }

    def load_model(self, symbol: str, timeframe: str, model_type: str) -> bool:
        """
        Load a trained model from disk.

        Args:
            symbol: Symbol the model was trained on
            timeframe: Timeframe the model was trained on
            model_type: Type of model to load

        Returns:
            True if model was loaded successfully, False otherwise
        """
        model_path = self._get_model_path(symbol, timeframe, model_type)

        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return False

        try:
            model_data = joblib.load(model_path)

            self.models[model_type] = {
                'model': model_data['model'],
                'feature_names': model_data['feature_names'],
                'metrics': model_data['metrics'],
                'prediction_horizon': model_data['prediction_horizon']
            }

            # Load scalers
            self.feature_generator.scalers = model_data['scalers']

            # Store current symbol and timeframe
            self.current_symbol = symbol
            self.current_timeframe = timeframe

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def predict(self,
                df: pd.DataFrame,
                model_type: str = 'random_forest',
                prediction_periods: int = 5) -> pd.DataFrame:
        """
        Make price predictions using a trained model.

        Args:
            df: DataFrame with OHLCV data
            model_type: Type of model to use for prediction
            prediction_periods: Number of periods to predict into the future

        Returns:
            DataFrame with predictions
        """
        if model_type not in self.models:
            raise ValueError(f"Model not loaded: {model_type}")

        model_data = self.models[model_type]
        model = model_data['model']
        feature_names = model_data['feature_names']
        prediction_horizon = model_data['prediction_horizon']

        # Generate features
        features_df = self.feature_generator.create_features(df)

        # Get the latest data point for prediction
        latest_data = features_df.iloc[-prediction_periods:].copy()

        # Initialize predictions DataFrame
        predictions = pd.DataFrame(index=range(prediction_periods))
        predictions['timestamp'] = [
            latest_data.iloc[-1]['timestamp'] + timedelta(hours=i + 1)
            if 'timestamp' in latest_data.columns
            else datetime.now() + timedelta(hours=i + 1)
            for i in range(prediction_periods)
        ]

        # Make recursive predictions
        current_features = latest_data.copy()

        for i in range(prediction_periods):
            # Extract features for the current prediction
            X = current_features.iloc[-1:][feature_names].values

            # Scale features if needed
            if 'features' in self.feature_generator.scalers:
                X = self.feature_generator.scalers['features'].transform(X)

            # Make prediction
            pred = model.predict(X)[0]

            # Store prediction
            predictions.loc[i, 'predicted_close'] = pred

            # If we need to make more predictions, create a new data point with the prediction
            if i < prediction_periods - 1:
                # Create a new row with the predicted value
                new_row = current_features.iloc[-1:].copy()
                new_row.index = [current_features.index[-1] + 1]

                # Update the close price with our prediction
                new_row['close'] = pred

                # Update other price columns with a reasonable estimate
                # (assuming the same price action as the previous candle)
                price_change = new_row['close'].values[0] - current_features.iloc[-1]['close']
                new_row['open'] = current_features.iloc[-1]['close']
                new_row['high'] = max(new_row['close'].values[0], new_row['open'].values[0])
                new_row['low'] = min(new_row['close'].values[0], new_row['open'].values[0])

                # Update timestamp
                if 'timestamp' in new_row.columns:
                    new_row['timestamp'] = current_features.iloc[-1]['timestamp'] + timedelta(hours=1)

                # Recalculate features for the new row
                # This is a simplified approach - in a real system, you'd need to handle this more carefully
                for col in new_row.columns:
                    if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                        new_row[col] = current_features.iloc[-1][col]

                # Append to current features
                current_features = pd.concat([current_features, new_row])

            # Add confidence intervals (simple approach)
        rmse = model_data['metrics']['rmse']
        predictions['lower_bound'] = predictions['predicted_close'] - 1.96 * rmse
        predictions['upper_bound'] = predictions['predicted_close'] + 1.96 * rmse

        # Add the last known price for reference
        predictions['last_known_close'] = df.iloc[-1]['close']

        return predictions

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the model if available."""
        if not hasattr(model, 'feature_importances_'):
            return None

        # Get feature importance
        importance = model.feature_importances_

        # Create a dictionary of feature importance
        return {feature_names[i]: importance[i] for i in range(len(feature_names))}

    def plot_predictions(self,
                         df: pd.DataFrame,
                         predictions: pd.DataFrame,
                         title: str = None) -> Figure:
        """
        Plot historical prices and predictions.

        Args:
            df: DataFrame with historical OHLCV data
            predictions: DataFrame with predictions from predict()
            title: Plot title

        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot historical prices
        historical_dates = df.index if not 'timestamp' in df.columns else df['timestamp']
        ax.plot(historical_dates[-30:], df['close'].values[-30:], label='Historical', color='blue')

        # Plot predictions
        prediction_dates = predictions['timestamp']
        ax.plot(prediction_dates, predictions['predicted_close'], label='Predicted', color='green', linestyle='--')

        # Plot confidence intervals
        ax.fill_between(
            prediction_dates,
            predictions['lower_bound'],
            predictions['upper_bound'],
            color='green',
            alpha=0.2,
            label='95% Confidence'
        )

        # Add last known price point to connect the lines
        ax.plot([historical_dates.iloc[-1], prediction_dates.iloc[0]],
                [df['close'].values[-1], predictions['predicted_close'].values[0]],
                color='green', linestyle='--')

        # Set title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{self.current_symbol} Price Prediction ({self.current_timeframe})")

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        # Format x-axis dates
        fig.autofmt_xdate()

        # Tight layout
        plt.tight_layout()

        return fig

    def evaluate_model(self,
                       df: pd.DataFrame,
                       model_type: str = 'random_forest',
                       plot: bool = True) -> Tuple[Dict[str, float], Optional[Figure]]:
        """
        Evaluate a trained model on historical data.

        Args:
            df: DataFrame with OHLCV data
            model_type: Type of model to evaluate
            plot: Whether to generate an evaluation plot

        Returns:
            Tuple of (metrics, figure)
        """
        if model_type not in self.models:
            raise ValueError(f"Model not loaded: {model_type}")

        model_data = self.models[model_type]
        model = model_data['model']
        feature_names = model_data['feature_names']
        prediction_horizon = model_data['prediction_horizon']

        # Generate features
        features_df = self.feature_generator.create_features(df)

        # Prepare data for evaluation
        X_train, X_test, y_train, y_test, _ = self.feature_generator.prepare_data_for_training(
            features_df,
            prediction_horizon=prediction_horizon,
            test_size=0.3
        )

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        # Create evaluation plot if requested
        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot actual vs predicted
            ax.scatter(y_test, y_pred, alpha=0.5)

            # Plot perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            # Set title and labels
            ax.set_title(f"{self.current_symbol} Model Evaluation ({model_type})")
            ax.set_xlabel('Actual Price')
            ax.set_ylabel('Predicted Price')

            # Add metrics as text
            metrics_text = f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}"
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})

            # Tight layout
            plt.tight_layout()

        return metrics, fig

class MLModelManager:
    """Manages multiple ML models for different symbols and timeframes."""

    def __init__(self, model_dir: str = "data/ml_models"):
        """
        Initialize the model manager.

        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = model_dir
        self.predictors = {}

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def get_predictor(self, symbol: str, timeframe: str) -> (
            PricePredictor):
        """
        Get a price predictor for a specific symbol and timeframe.

        Args:
            symbol: Symbol to predict
            timeframe: Timeframe to use

        Returns:
            PricePredictor instance
        """
        key = f"{symbol}_{timeframe}"

        if key not in self.predictors:
            self.predictors[key] = PricePredictor(self.model_dir)

        return self.predictors[key]

    def train_model_for_symbol(self,
                               df: pd.DataFrame,
                               symbol: str,
                               timeframe: str,
                               model_type: str = 'random_forest',
                               prediction_horizon: int = 1,
                               hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train a model for a specific symbol and timeframe.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol to predict
            timeframe: Timeframe to use
            model_type: Type of model to train
            prediction_horizon: Number of periods ahead to predict
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training results
        """
        predictor = self.get_predictor(symbol, timeframe)

        return predictor.train_model(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            prediction_horizon=prediction_horizon,
            hyperparameter_tuning=hyperparameter_tuning
        )

    def predict_for_symbol(self,
                           df: pd.DataFrame,
                           symbol: str,
                           timeframe: str,
                           model_type: str = 'random_forest',
                           prediction_periods: int = 5) -> pd.DataFrame:
        """
        Make predictions for a specific symbol and timeframe.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol to predict
            timeframe: Timeframe to use
            model_type: Type of model to use
            prediction_periods: Number of periods to predict

        Returns:
            DataFrame with predictions
        """
        predictor = self.get_predictor(symbol, timeframe)

        # Try to load the model if it's not already loaded
        if model_type not in predictor.models:
            success = predictor.load_model(symbol, timeframe, model_type)
            if not success:
                raise ValueError(f"Model not found for {symbol} ({timeframe}, {model_type})")

        return predictor.predict(
            df=df,
            model_type=model_type,
            prediction_periods=prediction_periods
        )

    def list_available_models(self) -> List[Dict[str, str]]:
        """
        List all available trained models.

        Returns:
            List of dictionaries with model information
        """
        models = []

        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            return models

        # List all files in the model directory
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.joblib'):
                # Parse filename to get symbol, timeframe, and model type
                parts = filename.replace('.joblib', '').split('_')

                if len(parts) >= 3:
                    symbol = parts[0]
                    timeframe = parts[1]
                    model_type = '_'.join(parts[2:])

                    models.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'model_type': model_type,
                        'filename': filename
                    })

        return models
