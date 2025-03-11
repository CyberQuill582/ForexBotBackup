import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class MLPredictor:
    def __init__(self, model_path="models", retrain_frequency=7):
        """
        Initialize ML predictor with model persistence

        Args:
            model_path: Directory to save/load models
            retrain_frequency: Days between model retraining (0 for manual only)
        """
        self.model_path = model_path
        self.retrain_frequency = retrain_frequency
        self.logger = logging.getLogger(__name__)
        self.feature_importance = None
        self.last_train_date = None

        # Initialize model and scaler
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        self.scaler = StandardScaler()

        # Create model directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Try to load existing model
        self._load_model()

    def get_feature_importance(self):
        """Return feature importance if available"""
        try:
            if self.model is not None and hasattr(self.model, 'feature_importances_'):
                return self.feature_importance
            else:
                self.logger.warning("Model not trained yet or feature importance not available")
                return pd.DataFrame({
                    'feature': ['No features yet'],
                    'importance': [0.0]
                })
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None

    def prepare_features(self, df):
        """
        Prepare features for ML model with enhanced feature engineering
        """
        try:
            features = pd.DataFrame()

            # Basic price-based features
            features['returns'] = df['Close'].pct_change()
            features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            features['volatility_20'] = df['Close'].rolling(window=20).std()
            features['volatility_50'] = df['Close'].rolling(window=50).std()

            # Technical indicators
            if 'RSI' in df.columns:
                features['rsi'] = df['RSI']
                features['rsi_diff'] = df['RSI'].diff()

            if 'MACD' in df.columns and 'Signal' in df.columns:
                features['macd'] = df['MACD']
                features['macd_signal'] = df['Signal']
                features['macd_diff'] = df['MACD'] - df['Signal']

            # Moving averages
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                features['sma_20'] = df['SMA_20']
                features['sma_50'] = df['SMA_50']
                features['sma_ratio'] = df['SMA_20'] / df['SMA_50']

            # Price patterns
            features['higher_high'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
            features['lower_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))

            # Volume-related features
            if 'Volume' in df.columns:
                features['volume'] = df['Volume']
                features['volume_ma20'] = df['Volume'].rolling(window=20).mean()
                features['volume_ratio'] = df['Volume'] / features['volume_ma20']

            # Range and candle features
            features['daily_range'] = (df['High'] - df['Low']) / df['Low']
            features['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
            features['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
            features['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']

            # Trend features
            features['price_up_trend'] = (df['Close'] > df['Close'].shift(1)).astype(int)
            features['consec_up'] = features['price_up_trend'].groupby(
                (features['price_up_trend'] != features['price_up_trend'].shift(1)).cumsum()
            ).cumcount() + 1

            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'close_lag_{lag}'] = df['Close'].shift(lag)
                features[f'return_lag_{lag}'] = features['returns'].shift(lag)

            # Remove NaN values
            features = features.dropna()

            return features
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise

    def predict(self, df, optimize=False, force_retrain=False):
        """
        Generate predictions using ML model with optional retraining
        """
        try:
            features = self.prepare_features(df)

            if len(features) < 100:  # Not enough data
                return np.zeros(len(df))

            # Check if we need to train/retrain the model
            should_train = force_retrain or self.model is None or not hasattr(self.model, 'estimators_')

            # Check training frequency if not forced
            if not should_train and self.last_train_date and self.retrain_frequency > 0:
                days_since_training = (datetime.now() - self.last_train_date).days
                should_train = days_since_training >= self.retrain_frequency

            if should_train:
                self.logger.info("Training ML model")
                self._train_model(features, optimize)
            else:
                self.logger.info("Using existing ML model")

            # Make predictions with existing model
            if not hasattr(self.model, 'estimators_'):
                self.logger.warning("Model not properly trained, returning zeros")
                return np.zeros(len(df))

            # Scale features for prediction
            features_scaled = self.scaler.transform(features)

            # Make predictions
            predictions = self.model.predict(features_scaled)

            # Pad predictions to match original data length
            full_predictions = np.zeros(len(df))
            full_predictions[-len(predictions):] = predictions

            return full_predictions
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            return np.zeros(len(df))

    def _train_model(self, features, optimize=False):
        """
        Train the ML model with provided features
        """
        try:
            # Create labels (next period returns)
            labels = np.where(features['returns'].shift(-1) > 0, 1, -1)[:-1]
            features = features[:-1]  # Remove last row to match labels

            # Train/test split with time series consideration
            train_size = int(len(features) * 0.8)
            X_train = features.iloc[:train_size]
            y_train = labels[:train_size]
            X_test = features.iloc[train_size:] if train_size < len(features) else features.iloc[train_size:]
            y_test = labels[train_size:] if train_size < len(labels) else []

            # Initialize scaler (already initialized in __init__)
            #self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Initialize or optimize model
            if optimize and len(X_train) > 200:
                self.logger.info("Performing hyperparameter optimization")
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train)
                self.model = RandomForestClassifier(random_state=42, **best_params)
            elif self.model is None:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

            # Calculate performance metrics if test data is available
            if len(y_test) > 0:
                X_test_scaled = self.scaler.transform(X_test)
                y_pred = self.model.predict(X_test_scaled)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                self.logger.info(f"Training metrics: {metrics}")

            # Update last training date
            self.last_train_date = datetime.now()

            # Save model
            self._save_model()

            return True
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False

    def optimize_hyperparameters(self, X, y):
        """
        Optimize model hyperparameters using grid search
        """
        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Grid search
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=tscv,
                scoring='f1_weighted',
                n_jobs=-1
            )

            grid_search.fit(X, y)

            self.logger.info(f"Best hyperparameters: {grid_search.best_params_}")

            return grid_search.best_params_
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return {}

    def _save_model(self):
        """Save model and scaler to disk"""
        try:
            model_file = os.path.join(self.model_path, "model.pkl")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.pkl")

            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)

            # Save scaler
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)

            # Save metadata
            metadata = {
                'last_train_date': self.last_train_date,
                'feature_importance': self.feature_importance
            }
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)

            self.logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def _load_model(self):
        """Load model and scaler from disk"""
        try:
            model_file = os.path.join(self.model_path, "model.pkl")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.pkl")

            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                self.logger.info("No existing model found, will train new model")
                return False

            # Load model
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)

            # Load scaler
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.last_train_date = metadata.get('last_train_date')
                self.feature_importance = metadata.get('feature_importance')

            self.logger.info(f"Model loaded from {self.model_path}")
            if self.last_train_date:
                self.logger.info(f"Last training date: {self.last_train_date}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False