import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class XGBoostMusicGenreClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, max_depth=12, learning_rate=0.05, random_state=42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=random_state,
            n_jobs=-1,
            enable_categorical=True
        )
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y):
        # Convert string labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        return self
        
    def predict(self, X):
        y_pred = self.model.predict(X)
        # Convert numeric predictions back to original string labels
        return self.label_encoder.inverse_transform(y_pred)
        
    def score(self, X, y):
        # Only score on labels that were seen during training
        y_encoded = []
        valid_indices = []
        for i, label in enumerate(y):
            if label in self.label_encoder.classes_:
                y_encoded.append(label)
                valid_indices.append(i)
        if not y_encoded:
            return 0.0  # No valid labels to score against
            
        # Get only the X values that correspond to valid labels
        X_valid = X[valid_indices]
        # Convert valid labels to encoded values and score
        y_encoded = self.label_encoder.transform(y_encoded)
        return self.model.score(X_valid, y_encoded)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
