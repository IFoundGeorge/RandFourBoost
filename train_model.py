# train_model.py
import os
import joblib
import numpy as np
from algorithm import MyCustomAlgorithm, load_dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def train_and_save_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare the dataset
    print("Loading dataset...")
    X, y = load_dataset("./dataset")  # Update this path to your dataset location
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training model...")
    model = MyCustomAlgorithm()
    model.fit(X_train_scaled, y_train)
    
    # Save the model and encoders
    joblib.dump(model, 'models/gentective_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print("Model and encoders saved successfully!")
    print(f"Model saved to: {os.path.abspath('models/gentective_model.pkl')}")

if __name__ == "__main__":
    train_and_save_model()