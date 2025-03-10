import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df):
        """
        Prepare features for ML model
        """
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['returns'] = df['Close'].pct_change()
            features['volatility'] = df['Close'].rolling(window=20).std()
            features['rsi'] = df['RSI']
            features['macd'] = df['MACD']
            features['macd_signal'] = df['Signal']
            features['sma_20'] = df['SMA_20']
            features['sma_50'] = df['SMA_50']
            
            # Remove NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def predict(self, df):
        """
        Generate predictions using ML model
        """
        try:
            features = self.prepare_features(df)
            
            if len(features) < 100:  # Not enough data
                return np.zeros(len(df))
            
            # Create labels (next day returns)
            labels = np.where(features['returns'].shift(-1) > 0, 1, -1)[:-1]
            
            # Train on most recent data
            train_size = int(len(features) * 0.8)
            X_train = features.iloc[:train_size]
            y_train = labels[:train_size]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(features)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = self.model.predict(X_test_scaled)
            
            # Pad predictions to match original data length
            full_predictions = np.zeros(len(df))
            full_predictions[-len(predictions):] = predictions
            
            return full_predictions
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            raise
