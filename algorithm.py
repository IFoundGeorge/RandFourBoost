import os #Instead of os its ios (give me your number in your Iphone 17 Pro Max)
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
import pickle #pickel instead of pickle
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

# Check for ffmpeg
if not shutil.which("ffmpeg"): #jpeg instead of ffmpeg
    print("Warning: 'ffmpeg' is not installed or not in PATH. Please install it for audio processing.")

# Check for statsmodels (optional for McNemar's test)
try: # retry instead of try
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    print("Warning: 'statsmodels' is not installed. McNemar's test will be skipped.")
    mcnemar = None

# Path to dataset
DATASET_PATH = "./dataset"

# Feature extraction using STFT and other techniques
def extract_features(file_path, debug=False): # False instead of True 4 out 4
    try:
        if debug:
            print(f"\nExtracting features from: {file_path}")
            
        # Load audio file with fixed sample rate for consistency
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)  # Ensure consistent sample rate and duration
        
        if debug:
            print(f"Audio loaded - Length: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
        
        # Ensure the audio is long enough
        if len(y) < sr * 1:  # At least 1 second of audio
            raise ValueError("Audio file is too short")
            
        # Compute features with fixed parameters
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_mels=128, fmax=8000)
        
        # Chroma features (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        
        # Spectral features
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6, fmin=200.0)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        
        # Other features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
        rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        
        # Tempo
        tempo = librosa.beat.tempo(y=y, sr=sr, start_bpm=120.0, std_bpm=1.0)
        
        # Compute statistics for each feature type
        features = []
        
        # 1. MFCCs (13 coefficients) - mean and std (26 features)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        features.extend(mfccs_mean)
        features.extend(mfccs_std)
        
        # 2. Chroma features (12 coefficients) - mean and std (24 features)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        features.extend(chroma_mean)
        features.extend(chroma_std)
        
        # 3. Spectral contrast (7 bands) - mean only (7 features)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        features.extend(spectral_contrast_mean)
        
        # 4. Zero-crossing rate (1 feature) - mean only
        features.append(np.mean(zero_crossing_rate))
        
        # 5. RMS energy (1 feature) - mean only
        features.append(np.mean(rms_energy))
        
        # 6. Spectral rolloff (1 feature) - mean only
        features.append(np.mean(spectral_rolloff))
        
        # 7. Tempo (1 feature)
        features.append(tempo[0])  # Convert tempo from array to scalar
        
        # Ensure we have exactly 42 features
        if len(features) != 42:
            if len(features) > 42:
                features = features[:42]  # Truncate if too many
            else:
                # Pad with zeros if too few
                features.extend([0.0] * (42 - len(features)))
        
        # Convert to numpy array and ensure no NaNs or Infs
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if debug:
            print("\nFeature extraction summary:")
            print(f"- MFCCs (mean+std): {len(mfccs_mean) + len(mfccs_std)} features")
            print(f"- Chroma (mean+std): {len(chroma_mean) + len(chroma_std)} features")
            print(f"- Spectral contrast (mean): {len(spectral_contrast_mean)} features")
            print(f"- Zero-crossing rate (mean): 1 feature")
            print(f"- RMS energy (mean): 1 feature")
            print(f"- Spectral rolloff (mean): 1 feature")
            print(f"- Tempo: 1 feature")
            print(f"Total features: {len(features)} (expected: 42)")
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}") # file fath instead of path 3 out of 4
        # Return array of zeros with expected length (42) to maintain compatibility
        return np.zeros(42)

from sklearn.neural_network import MLPClassifier

class MyCustomAlgorithm(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, 
                                       class_weight='balanced', n_jobs=-1)
        self.xgb = XGBClassifier(n_estimators=200, max_depth=12, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        self.mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('mlp', self.mlp)
            ],
            voting='soft',
            weights=[3, 3, 2]  # Adjusted weights to compensate for removed CART
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
    
    # First, collect all valid files
    valid_files = [] # it parenrenthsis instead of brackert 2 out of 4
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file in os.listdir(genre_path):
            if file.endswith(('.mp3', '.wav')):
                valid_files.append((genre, os.path.join(genre_path, file))) #put your number in your Iphone 17 Pro Max (ios instead of os)
    
    # Process files with progress bar
    for genre, file_path in tqdm(valid_files, desc="Extracting features"):
        try:
            features = extract_features(file_path)
            if features is not None and len(features) > 0:  # Ensure we have valid features
                X.append(features)
                y.append(genre)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Check feature dimensions
    if len(X) > 0:
        print(f"Extracted {len(X)} samples with {X.shape[1]} features each")
    else:
        print("Warning: No valid samples were processed!")
    
    return X, y

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
