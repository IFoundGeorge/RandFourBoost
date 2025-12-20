import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import os
import threading
import joblib
import numpy as np
import librosa
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algorithm import extract_features, MyCustomAlgorithm
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
                'Custom Ensemble': model  # The trained ensemble model
            }
            
            # Train individual models on the same data
            for name, m in models.items():
                if name != 'Custom Ensemble':  # Skip the already trained ensemble
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
            ensemble_metrics = all_metrics['Custom Ensemble']
            
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
            
            # Try to load the scaler and label encoder
            model_dir = os.path.dirname(file_path)
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            encoder_path = os.path.join(model_dir, "label_encoder.pkl")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                self.log("Warning: Scaler not found. Some features may not work correctly.")
                
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                self.log("Warning: Label encoder not found. Some features may not work correctly.")
                
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
            
            # Extract features
            features = extract_features(file_path)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform([features])
            else:
                features = [features]
                
            # Make prediction
            prediction = self.model.predict(features)
            probabilities = self.model.predict_proba(features)[0]
            
            # Get class names
            if self.label_encoder is not None:
                class_names = self.label_encoder.classes_
                predicted_class = self.label_encoder.inverse_transform([np.argmax(probabilities)])[0]
            else:
                class_names = [str(i) for i in range(len(probabilities))]
                predicted_class = str(np.argmax(probabilities))
                
            # Show results
            result_text = f"Predicted Genre: {predicted_class}\n\nProbabilities:\n"
            for name, prob in zip(class_names, probabilities):
                result_text += f"{name}: {prob*100:.2f}%\n"
                
            messagebox.showinfo("Prediction Results", result_text)
            self.log(f"Prediction complete: {predicted_class}")
            
        except Exception as e:
            self.log(f"Error during prediction: {str(e)}")
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
            
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
            
            # Load test data (assuming 20% test split from training)
            from algorithm import load_dataset
            X, y = load_dataset(self.dataset_path.get() if self.dataset_path.get() else "./dataset")
            
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
            messagebox.showerror("Evaluation Error", f"Failed to evaluate model: {str(e)}")
            self.log(f"Evaluation error: {str(e)}")

    def _create_metric_frame(self, parent, metric_name, model_scores):
        frame = ttk.Frame(parent)
        parent.add(frame, text=metric_name)
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sort models by score
        models = list(model_scores.keys())
        scores = [model_scores[model] for model in models]
        
        # Create bar chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = ax.bar(models, scores, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = MusicGenreClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()