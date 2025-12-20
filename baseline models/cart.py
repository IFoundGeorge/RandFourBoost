import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

class CARTMusicGenreClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=15, random_state=42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
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
        for label in y:
            if label in self.label_encoder.classes_:
                y_encoded.append(label)
        if not y_encoded:
            return 0.0  # No valid labels to score against
            
        # Convert valid labels to encoded values and score
        y_encoded = self.label_encoder.transform(y_encoded)
        return self.model.score(X[:len(y_encoded)], y_encoded)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
