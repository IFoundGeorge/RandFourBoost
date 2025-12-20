import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay, log_loss, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shutil
import pickle
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

# Check for ffmpeg
if not shutil.which("ffmpeg"):
    print("Warning: 'ffmpeg' is not installed or not in PATH. Please install it for audio processing.")

# Check for statsmodels (optional for McNemar's test)
try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    print("Warning: 'statsmodels' is not installed. McNemar's test will be skipped.")
    mcnemar = None

# Path to dataset
DATASET_PATH = "./dataset"

# Feature extraction using STFT and other techniques
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        stft = np.abs(librosa.stft(y))

        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(stft), sr=sr, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).flatten()
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()
        rms_energy = librosa.feature.rms(y=y).flatten()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(zero_crossing_rate),
            np.mean(chroma, axis=1),
            np.mean(spectral_rolloff),
            np.mean(rms_energy),
            tempo,
            np.mean(tonnetz, axis=1)
        ))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

from sklearn.neural_network import MLPClassifier

class MyCustomAlgorithm(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.cart = DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42)
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        self.xgb = XGBClassifier(n_estimators=200, max_depth=12, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

        self.ensemble = VotingClassifier(
            estimators=[
                ('cart', self.cart),
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('mlp', self.mlp)
            ],
            voting='soft',
            weights=[1, 2, 3, 2]
        )

    def fit(self, X, y):
        self.ensemble.fit(X, y)
        return self

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)


# Load dataset
def load_dataset(dataset_path):
    genres = [g for g in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, g))]
    X, y = [], []

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            file_path = os.path.join(genre_path, file)
            if file.endswith(('.mp3', '.wav')):
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(genre)

    return np.array(X), np.array(y)

# Main pipeline
def main():
    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    if len(X) == 0:
        print("No valid audio files found. Exiting...")
        return

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Save the splits
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('saved_data', exist_ok=True)
    
    # Save the splits
    with open(f'saved_data/X_train_{timestamp}.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open(f'saved_data/X_test_{timestamp}.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(f'saved_data/y_train_{timestamp}.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(f'saved_data/y_test_{timestamp}.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    # Save label encoder
    with open(f'saved_data/label_encoder_{timestamp}.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f'saved_data/scaler_{timestamp}.joblib')

    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Models Dictionary
    models = {
        'CART': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'My Custom Algorithm': MyCustomAlgorithm()
    }

    # Train the models
    print("Training models...")
    for name, model in models.items():
        if name == 'Voting':
            print("Training Voting Classifier...")
            voting_model = VotingClassifier(estimators=[
                ('cart', models['CART']),
                ('rf', models['Random Forest']),
                ('xgb', models['XGBoost'])
            ], voting='soft', weights=[1, 2, 3])
            models['Voting'] = voting_model
            models['Voting'].fit(X_train_resampled, y_train_resampled)
        else:
            print(f"Training {name}...")
            model.fit(X_train_resampled, y_train_resampled)

    # After training all models, save them
    os.makedirs('saved_models', exist_ok=True)
    for name, model in models.items():
        model_name = name.lower().replace(' ', '_')
        joblib.dump(model, f'saved_models/{model_name}_{timestamp}.joblib')
    
    print(f"\nAll models and data splits have been saved with timestamp: {timestamp}")
    print(f"Data splits saved to: saved_data/")
    print(f"Models saved to: saved_models/")

    # Evaluate all models
    print("Evaluating models...")
    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_scaled)
            auc = roc_auc_score(y_test, y_score, multi_class='ovr')
            logloss = log_loss(y_test, y_score)
            results[name] = {
                'AUC (OvR)': auc,
                'Logarithmic Loss': logloss,
                'Percent Correct Classification': accuracy * 100,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
            print(f"{name} AUC (OvR): {auc:.2f}")
            print(f"{name} Logarithmic Loss: {logloss:.2f}")
        else:
            results[name] = {
                'Percent Correct Classification': accuracy * 100,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }

        print(f"{name} Percent Correct Classification: {accuracy * 100:.2f}%")
        print(f"{name} Precision: {precision:.2f}")
        print(f"{name} Recall: {recall:.2f}")
        print(f"{name} F1 Score: {f1:.2f}")

    # Summary of results
    print("\nSummary of Results:")
    for name, result in results.items():
        print(f"{name}: AUC: {result.get('AUC (OvR)', 'N/A')}, "
              f"Logarithmic Loss: {result.get('Logarithmic Loss', 'N/A')}, "
              f"Percent Correct Classification: {result['Percent Correct Classification']:.2f}%, "
              f"Precision: {result['Precision']:.2f}, "
              f"Recall: {result['Recall']:.2f}, "
              f"F1 Score: {result['F1 Score']:.2f}")

if __name__ == "__main__":
    main()
