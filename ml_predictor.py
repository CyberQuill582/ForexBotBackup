import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.feature_importance = None
    
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
            features['rsi'] = df['RSI']
            features['rsi_diff'] = df['RSI'].diff()
            features['macd'] = df['MACD']
            features['macd_signal'] = df['Signal']
            features['macd_diff'] = df['MACD'] - df['Signal']
            
            # Moving averages
            features['sma_20'] = df['SMA_20']
            features['sma_50'] = df['SMA_50']
            features['sma_ratio'] = df['SMA_20'] / df['SMA_50']
            
            # Price patterns
            features['higher_high'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
            features['lower_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
            
            # Volume-related features (if available)
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
            features['consec_up'] = features['price_up_trend'].groupby((features['price_up_trend'] != features['price_up_trend'].shift(1)).cumsum()).cumcount() + 1
            
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
    
    def predict(self, df, optimize=False):
        """
        Generate predictions using ML model with optional hyperparameter optimization
        """
        try:
            features = self.prepare_features(df)
            
            if len(features) < 100:  # Not enough data
                return np.zeros(len(df))
            
            # Create labels (next period returns)
            labels = np.where(features['returns'].shift(-1) > 0, 1, -1)[:-1]
            
            # Train/test split with time series consideration
            train_size = int(len(features) * 0.8)
            X_train = features.iloc[:train_size]
            y_train = labels[:train_size]
            X_test = features.iloc[train_size:-1] if train_size < len(features)-1 else features.iloc[train_size:]
            y_test = labels[train_size:] if train_size < len(labels) else []
            
            # Hyperparameter optimization
            if optimize and len(X_train) > 200:
                self.logger.info("Performing hyperparameter optimization")
                self.optimize_hyperparameters(X_train, y_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(features.iloc[:-1] if len(features) > 1 else features)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            # Make predictions
            predictions = self.model.predict(X_test_scaled)
            
            # Calculate performance metrics if test data is available
            if len(y_test) > 0:
                y_pred = predictions[-len(y_test):]
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                self.logger.info(f"Prediction metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Pad predictions to match original data length
            full_predictions = np.zeros(len(df))
            full_predictions[-len(predictions):] = predictions
            
            return full_predictions
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, X, y):
        """
        Optimize model hyperparameters using grid search and time series cross-validation
        """
        try:
            # Define parameter grid for RandomForest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            # Grid search with time series cross-validation
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid={'classifier__' + key: val for key, val in param_grid.items()},
                cv=tscv,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
            self.model = RandomForestClassifier(random_state=42, **best_params)
            
            self.logger.info(f"Best hyperparameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """Return feature importance if available"""
        return self.feature_importance
