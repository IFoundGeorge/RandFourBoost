import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import os
import threading
import joblib
import numpy as np
import librosa
from datetime import datetime
import shutil
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from algorithm import extract_features, MyCustomAlgorithm
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import pandas as pd
import pickle

class MusicGenreClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Genre Classifier")
        self.root.geometry("800x600")
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.training_in_progress = False
        
        # Configure styles
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=5)
        
        # Create main frames
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset", padding="10")
        dataset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dataset_frame, text="Dataset Path:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(dataset_frame, text="Browse...", command=self.browse_dataset).pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Train Model", command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Predict File", command=self.predict_file).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Log text area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(self.log_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            self.dataset_path.set(path)
            self.log(f"Selected dataset: {path}")
            
    def start_training(self):
        if not self.dataset_path.get() or not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Error", "Please select a valid dataset folder")
            return
            
        if self.training_in_progress:
            messagebox.showinfo("Info", "Training is already in progress")
            return
            
        self.training_in_progress = True
        self.log("Starting model training...")
        
        # Start training in a separate thread to keep the GUI responsive
        training_thread = threading.Thread(target=self.train_model)
        training_thread.daemon = True
        training_thread.start()
        
    def train_model(self):
        try:
            # Import here to avoid circular imports
            from algorithm import load_dataset, MyCustomAlgorithm
            from sklearn.model_selection import train_test_split
            
            # Load dataset
            self.log("Loading dataset...")
            X, y = load_dataset(self.dataset_path.get())
            
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Train model
            self.log("Training model...")
            model = MyCustomAlgorithm()
            model.fit(X_train_scaled, y_train_encoded)
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/ranboost_fourcart_model_{timestamp}.joblib"
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)
            
            # Save scaler and label encoder
            joblib.dump(self.scaler, f"models/scaler_{timestamp}.joblib")
            with open(f"models/label_encoder_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            self.model = model
            self.log(f"Model trained and saved to {model_path}")
            
            # Evaluate model
            self.log("Evaluating models...")
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize individual models
            from algorithm import DecisionTreeClassifier, RandomForestClassifier, XGBClassifier
            
            models = {
                'CART': DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
                'XGBoost': XGBClassifier(n_estimators=200, max_depth=12, learning_rate=0.05,
                                      use_label_encoder=False, eval_metric='mlogloss', random_state=42),
                'Ranboost Fourcart': model  # The trained ensemble model
            }
            
            # Train individual models on the same data
            for name, m in models.items():
                if name != 'Ranboost Fourcart':  # Skip the already trained ensemble
                    self.log(f"Training {name} for comparison...")
                    m.fit(X_train_scaled, y_train_encoded)
            
            # Calculate metrics for all models
            from sklearn.metrics import (
                accuracy_score, log_loss, roc_auc_score, classification_report,
                precision_score, recall_score, f1_score, confusion_matrix
            )
            import pandas as pd
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create evaluation window
            eval_window = Toplevel(self.root)
            eval_window.title("Model Evaluation Results")
            eval_window.geometry("1200x900")
            
            # Notebook for tabs
            notebook = ttk.Notebook(eval_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 1. Model Comparison Tab
            comparison_frame = ttk.Frame(notebook)
            notebook.add(comparison_frame, text="Model Comparison")
            
            # Calculate metrics for all models
            all_metrics = {}
            for name, m in models.items():
                y_pred = m.predict(X_test_scaled)
                y_pred_proba = m.predict_proba(X_test_scaled) if hasattr(m, 'predict_proba') else None
                
                metrics = {
                    'Accuracy': accuracy_score(y_test_encoded, y_pred),
                    'Precision (Macro)': precision_score(y_test_encoded, y_pred, average='macro', zero_division=0),
                    'Recall (Macro)': recall_score(y_test_encoded, y_pred, average='macro'),
                    'F1 Score (Macro)': f1_score(y_test_encoded, y_pred, average='macro'),
                    'Precision (Weighted)': precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
                    'Recall (Weighted)': recall_score(y_test_encoded, y_pred, average='weighted'),
                    'F1 Score (Weighted)': f1_score(y_test_encoded, y_pred, average='weighted')
                }
                
                if y_pred_proba is not None:
                    try:
                        metrics['Log Loss'] = log_loss(y_test_encoded, y_pred_proba)
                        if len(self.label_encoder.classes_) == 2:
                            metrics['AUC'] = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                        else:
                            metrics['AUC (OvR)'] = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
                            metrics['AUC (OvO)'] = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovo')
                    except Exception as e:
                        self.log(f"Could not calculate some metrics for {name}: {str(e)}")
                
                all_metrics[name] = metrics
            
            # Create a comparison table
            comparison_df = pd.DataFrame.from_dict(all_metrics, orient='index')
            
            # Create a text widget to display the comparison
            comparison_text = tk.Text(comparison_frame, wrap=tk.NONE, font=('Courier', 10))
            comparison_text.insert(tk.END, comparison_df.to_string())
            comparison_text.config(state=tk.DISABLED)
            
            # Add scrollbars
            vsb = ttk.Scrollbar(comparison_frame, orient="vertical", command=comparison_text.yview)
            hsb = ttk.Scrollbar(comparison_frame, orient="horizontal", command=comparison_text.xview)
            comparison_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            vsb.pack(side="right", fill="y")
            hsb.pack(side="bottom", fill="x")
            comparison_text.pack(fill="both", expand=True)
            
            # 2. Overall Metrics Tab (for the main ensemble model)
            metrics_frame = ttk.Frame(notebook)
            notebook.add(metrics_frame, text="Ensemble Metrics")
            
            # Get metrics for the ensemble model
            ensemble_metrics = all_metrics['Ranboost Fourcart']
            
            # Display metrics in a table
            metrics_text = ""
            for metric, value in ensemble_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_text += f"{metric}: {value:.4f}\n"
                else:
                    metrics_text += f"{metric}: {value}\n"
            
            metrics_label = ttk.Label(metrics_frame, text=metrics_text, justify=tk.LEFT, font=('Courier', 10))
            metrics_label.pack(padx=10, pady=10, anchor='nw')
            
            # 2. Classification Report Tab
            report_frame = ttk.Frame(notebook)
            notebook.add(report_frame, text="Classification Report")
            
            # Generate classification report
            report = classification_report(y_test_encoded, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            # Create a text widget to display the report
            report_text = tk.Text(report_frame, wrap=tk.NONE, font=('Courier', 10))
            report_text.insert(tk.END, report_df.to_string())
            report_text.config(state=tk.DISABLED)
            
            # Add scrollbars
            vsb = ttk.Scrollbar(report_frame, orient="vertical", command=report_text.yview)
            hsb = ttk.Scrollbar(report_frame, orient="horizontal", command=report_text.xview)
            report_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            
            vsb.pack(side="right", fill="y")
            hsb.pack(side="bottom", fill="x")
            report_text.pack(fill="both", expand=True)
            
            # 3. Confusion Matrix Tab
            cm_frame = ttk.Frame(notebook)
            notebook.add(cm_frame, text="Confusion Matrix")
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues)
            plt.colorbar(cax)
            
            # Set labels
            ax.set_xticks(np.arange(len(self.label_encoder.classes_)))
            ax.set_yticks(np.arange(len(self.label_encoder.classes_)))
            ax.set_xticklabels(self.label_encoder.classes_, rotation=45, ha='left')
            ax.set_yticklabels(self.label_encoder.classes_)
            
            # Add text annotations
            for i in range(len(self.label_encoder.classes_)):
                for j in range(len(self.label_encoder.classes_)):
                    ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                           ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix (Counts and Normalized)')
            
            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=cm_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 4. ROC Curve Tab - Use the dedicated method
            self._create_roc_curve(notebook, y_test_encoded, y_pred_proba)
            
            self.log("Model training and evaluation completed successfully!")
            
        except Exception as e:
            self.log(f"Error during training: {str(e)}")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
        finally:
            self.training_in_progress = False
            
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.log(f"Loading model from {file_path}...")
            self.model = joblib.load(file_path)
            
            # Get the model directory and base name
            model_dir = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Look for label encoder and scaler in the same directory
            label_encoder_path = None
            scaler_path = None
            
            # First try to find files with the same timestamp
            if '_' in base_name:
                timestamp = '_'.join(base_name.split('_')[-2:])
                for f in os.listdir(model_dir):
                    if f.startswith('label_encoder_') and timestamp in f and f.endswith('.pkl'):
                        label_encoder_path = os.path.join(model_dir, f)
                    elif f.startswith('scaler_') and timestamp in f and f.endswith('.joblib'):
                        scaler_path = os.path.join(model_dir, f)
            
            # If not found, try to find any label encoder and scaler in the directory
            if label_encoder_path is None or not os.path.exists(label_encoder_path):
                for f in os.listdir(model_dir):
                    if f.startswith('label_encoder_') and f.endswith('.pkl'):
                        label_encoder_path = os.path.join(model_dir, f)
                        break
            
            if scaler_path is None or not os.path.exists(scaler_path):
                for f in os.listdir(model_dir):
                    if f.startswith('scaler_') and f.endswith('.joblib'):
                        scaler_path = os.path.join(model_dir, f)
                        break
            
            # Load the label encoder if found
            if label_encoder_path and os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = joblib.load(f)
                self.log(f"Loaded label encoder: {os.path.basename(label_encoder_path)}")
                
                # Log the class names for debugging
                if hasattr(self.label_encoder, 'classes_'):
                    self.log(f"Available genres: {', '.join(self.label_encoder.classes_)}")
            else:
                self.log("Warning: No label encoder found. Predictions will show numeric labels.")
            
            # Load the scaler if found
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.log(f"Loaded scaler: {os.path.basename(scaler_path)}")
            else:
                self.log("Warning: No scaler found. Predictions may be inaccurate.")
            
            # Log model information
            if hasattr(self.model, 'n_features_in_'):
                self.log(f"Model expects {self.model.n_features_in_} features")
            
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("Error", error_msg)
            if self.scaler is None and os.path.exists('saved_models/scaler.joblib'):
                self.scaler = joblib.load('saved_models/scaler.joblib')
                self.log("Loaded default scaler")
                
            if self.label_encoder is None and os.path.exists('saved_models/label_encoder.pkl'):
                with open('saved_models/label_encoder.pkl', 'rb') as f:
                    self.label_encoder = pickle.load(f)
                self.log("Loaded default label encoder")
            
            if self.scaler is None:
                self.log("Warning: No scaler found. Predictions may be inaccurate.")
            if self.label_encoder is None:
                self.log("Warning: No label encoder found. Class labels may be incorrect.")
                
            # Log model and scaler information
            if hasattr(self.model, 'n_features_in_'):
                self.log(f"Model expects {self.model.n_features_in_} features")
            if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
                self.log(f"Scaler expects {self.scaler.n_features_in_} features")
                
            self.log("Model loaded successfully!")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def predict_file(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load or train a model first")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.ogg"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.log(f"Processing {os.path.basename(file_path)}...")
            
            # Extract features with debug mode on
            features = extract_features(file_path, debug=True)
            
            if features is None or len(features) == 0:
                raise ValueError("Failed to extract features from the audio file")
                
            # Ensure features are in the right shape (1, n_features)
            features = np.array(features).reshape(1, -1)
            
            # Log the number of features extracted
            self.log(f"Extracted {features.shape[1]} features")
            
            # Log model's expected features if available
            if hasattr(self.model, 'n_features_in_'):
                self.log(f"Model expects {self.model.n_features_in_} features")
            elif hasattr(self.model, 'estimators_'):
                # For ensemble models, check the first estimator
                first_estimator = self.model.estimators_[0] if hasattr(self.model, 'estimators_') else None
                if first_estimator and hasattr(first_estimator, 'n_features_in_'):
                    self.log(f"First estimator expects {first_estimator.n_features_in_} features")
            
            # If we have a scaler, check dimensions and scale
            if self.scaler is not None:
                self.log(f"Scaler expects {self.scaler.n_features_in_} features")
                
                # If dimensions don't match, try to pad or truncate
                if features.shape[1] != self.scaler.n_features_in_:
                    self.log(f"Adjusting features from {features.shape[1]} to {self.scaler.n_features_in_}")
                    
                    # Create a new array with the expected number of features
                    adjusted_features = np.zeros((1, self.scaler.n_features_in_))
                    
                    # Copy available features, truncating or padding with zeros as needed
                    min_len = min(features.shape[1], self.scaler.n_features_in_)
                    adjusted_features[0, :min_len] = features[0, :min_len]
                    
                    features = adjusted_features
                
                # Scale the features
                features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features)
            
            # Get probabilities if the model supports it
            try:
                probabilities = self.model.predict_proba(features)[0]
                has_probabilities = True
            except (AttributeError, NotImplementedError):
                has_probabilities = False
            
            # Get class names and handle prediction
            if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
                try:
                    # Get the predicted class name
                    predicted_class = self.label_encoder.inverse_transform(prediction)[0]
                    # Get all class names
                    class_names = self.label_encoder.classes_
                    
                    # Show results
                    result_text = f"Predicted Genre: {predicted_class}\n\n"
                    
                    if has_probabilities:
                        result_text += "Confidence Scores:\n"
                        # Sort probabilities in descending order
                        sorted_indices = np.argsort(probabilities)[::-1]
                        for idx in sorted_indices:
                            genre = class_names[idx]
                            prob = probabilities[idx] * 100
                            result_text += f"{genre}: {prob:.1f}%\n"
                    else:
                        result_text += "(Probability scores not available for this model)\n"
                    
                    messagebox.showinfo("Prediction Results", result_text)
                    self.log(f"Prediction complete: {predicted_class}")
                    
                except Exception as e:
                    error_msg = f"Error decoding prediction: {str(e)}"
                    self.log(error_msg)
                    messagebox.showerror("Prediction Error", error_msg)
            else:
                # Fallback if no label encoder is available
                predicted_class = str(prediction[0])
                result_text = f"Predicted Class: {predicted_class}\n"
                if has_probabilities:
                    result_text += "\nConfidence Scores:\n"
                    for i, prob in enumerate(probabilities):
                        result_text += f"Class {i}: {prob*100:.1f}%\n"
                
                messagebox.showinfo("Prediction Results", result_text)
                self.log(f"Prediction complete (numeric class): {predicted_class}")
            
        except Exception as e:
            error_msg = f"Failed to make prediction: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("Prediction Error", error_msg)
            
    def show_evaluation_metrics(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load or train a model first")
            return
            
        try:
            # Create a new window
            eval_window = Toplevel(self.root)
            eval_window.title("Model Evaluation Metrics")
            eval_window.geometry("1000x600")
            
            # Create a notebook for different metrics
            notebook = ttk.Notebook(eval_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            try:
                # Load test data (assuming 20% test split from training)
                from algorithm import load_dataset
                X, y = load_dataset(self.dataset_path.get() if self.dataset_path.get() else "./dataset")
                
                if len(X) == 0 or len(y) == 0:
                    raise ValueError("No data loaded. Please check the dataset path.")
                
                # Split data (80% train, 20% test)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features if scaler is available
                if self.scaler is not None:
                    X_test_scaled = self.scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Make predictions
                y_pred = self.model.predict(X_test_scaled)
                y_pred_proba = self.model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score, log_loss, roc_auc_score,
                    precision_score, recall_score, f1_score
                )
                
                # For multi-class classification, we need to specify multi_class='ovr' and average='weighted'
                accuracy = accuracy_score(y_test, y_pred)
                logloss = log_loss(y_test, y_pred_proba)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Calculate AUC (handle multi-class)
                try:
                    # For binary classification
                    if len(self.label_encoder.classes_) == 2:
                        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        # For multi-class (one-vs-rest)
                        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except Exception as e:
                    self.log(f"Could not calculate AUC: {str(e)}")
                    auc = 0.0
                
                # Create metric frames
                self._create_metric_frame(notebook, "Accuracy", {"Trained Model": accuracy})
                self._create_metric_frame(notebook, "Log Loss", {"Trained Model": logloss})
                self._create_metric_frame(notebook, "AUC (OvR)", {"Trained Model": auc})
                self._create_metric_frame(notebook, "Precision", {"Trained Model": precision})
                self._create_metric_frame(notebook, "Recall", {"Trained Model": recall})
                self._create_metric_frame(notebook, "F1 Score", {"Trained Model": f1})
                
                # Create ROC Curve tab
                self._create_roc_curve(notebook, y_test, y_pred_proba)
                
                # Create ROC Curve tab
                self._create_roc_curve(notebook, y_test, y_pred_proba)
                
                # Summary frame
                summary_frame = ttk.Frame(notebook)
                notebook.add(summary_frame, text="Summary")
                
                # Summary text
                summary_text = f"""Model Evaluation Summary:
                
    Trained Model Performance:
    - Accuracy: {accuracy:.4f}
    - Log Loss: {logloss:.4f}
    - AUC (OvR): {auc:.4f}
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 Score: {f1:.4f}

    Evaluation on {len(X_test)} test samples
    """
                
                text_widget = tk.Text(summary_frame, wrap=tk.WORD, padx=10, pady=10, font=('Consolas', 10))
                text_widget.insert(tk.END, summary_text)
                text_widget.pack(fill=tk.BOTH, expand=True)
                text_widget.config(state=tk.DISABLED)
                
                # Log the evaluation
                self.log(f"Model evaluation completed on {len(X_test)} test samples")
                
            except Exception as e:
                # Show error in the window if dataset loading fails
                error_frame = ttk.Frame(notebook)
                notebook.add(error_frame, text="Error")
                error_label = ttk.Label(
                    error_frame, 
                    text=f"Error loading dataset: {str(e)}\n\nPlease make sure the dataset path is correct and contains valid audio files.",
                    wraplength=800,
                    padding=20
                )
                error_label.pack(expand=True)
                self.log(f"Evaluation error: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create evaluation window: {str(e)}")
            self.log(f"Error in show_evaluation_metrics: {str(e)}")

    def _create_metric_frame(self, notebook, metric_name, model_scores):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=metric_name)
        
        # Create a treeview to display the scores
        tree = ttk.Treeview(frame, columns=('Model', 'Score'), show='headings')
        tree.heading('Model', text='Model')
        tree.heading('Score', text='Score')
        
        # Insert data
        for model_name, score in model_scores.items():
            tree.insert('', 'end', values=(model_name, f"{score:.4f}"))
            
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def _create_roc_curve(self, notebook, y_test, y_pred_proba):
        try:
            # Create a new tab for ROC curve
            roc_frame = ttk.Frame(notebook)
            notebook.add(roc_frame, text="ROC Curve")
            
            # Create a figure for the ROC curve
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Binarize the output for multi-class ROC
            n_classes = len(self.label_encoder.classes_)
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            # Calculate ROC for each class
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Plot ROC curves
            colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
            
            # Plot each class ROC curve
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=1,
                        label='{0} (AUC = {1:0.2f})'.format(self.label_encoder.classes_[i], roc_auc[i]))
            
            # Plot micro-average ROC curve
            ax.plot(fpr["micro"], tpr["micro"],
                    label='micro-avg (AUC = {0:0.2f})'.format(roc_auc["micro"]),
                    color='deeppink', linestyle=':', linewidth=2)
            
            # Plot random guessing line
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            
            # Set plot properties
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve (One-vs-Rest)')
            
            # Adjust legend to be outside the plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
            
            # Create a canvas and add it to the frame
            canvas = FigureCanvasTkAgg(fig, master=roc_frame)
            canvas.draw()
            
            # Pack the canvas
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create a toolbar for the figure
            toolbar = NavigationToolbar2Tk(canvas, roc_frame)
            toolbar.update()
            
            # Adjust layout
            fig.tight_layout()
            
        except Exception as e:
            self.log(f"Error creating ROC curve: {str(e)}")
            messagebox.showerror("Error", f"Failed to create ROC curve: {str(e)}")

def main():
    root = tk.Tk()
    app = MusicGenreClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()