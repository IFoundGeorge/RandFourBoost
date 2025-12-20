import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import os
import threading
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import numpy as np
import librosa
from datetime import datetime
import shutil
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from algorithm import extract_features, MyCustomAlgorithm
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import pandas as pd
import pickle
import time

class GentectiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gentective")
        self.root.geometry("1200x800")  # Larger initial size for better banner display
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Bind the resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Variables
        self.dataset_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.training_in_progress = False
        self.current_banner = 1
        self.slideshow_running = False
        self.slideshow_interval = 3000  # 3 seconds between slides
        
        # Configure styles
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=5)
        
        # Set up the banner first
        self.setup_banner()
        
        # Then set up the main UI
        self.setup_ui()
        
        # Start the slideshow
        self.start_slideshow()
        
        # Bind window close event to stop slideshow
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def setup_banner(self):
        """Set up the banner at the top of the application"""
        # Create banner frame at the very top of the window with no padding/margin
        self.banner_frame = ttk.Frame(self.root)
        self.banner_frame.pack(fill=tk.X, expand=False, side=tk.TOP, padx=0, pady=0)
        
        # Set a fixed height for the banner and prevent content from affecting size
        self.banner_height = 200  # Store as instance variable for easy access
        self.banner_frame.config(height=self.banner_height)
        self.banner_frame.pack_propagate(False)
        
        # Create a canvas for the banner that will stretch to fill the frame
        self.banner_canvas = tk.Canvas(
            self.banner_frame, 
            highlightthickness=0, 
            bd=0,
            width=self.root.winfo_width(),  # Use window width instead of screen width
            height=self.banner_height,
            bg='#f0f0f0'  # Light gray background in case no image loads
        )
        self.banner_canvas.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Add horizontal scrollbar (will only show when needed)
        self.scrollbar = ttk.Scrollbar(self.banner_frame, orient=tk.HORIZONTAL, command=self.banner_canvas.xview)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.banner_canvas.configure(
            xscrollcommand=self.scrollbar.set,
            scrollregion=(0, 0, self.root.winfo_screenwidth() * 2, 200)  # Wider scrollable area
        )
        
        # Create a frame inside the canvas to hold the image
        self.banner_container = ttk.Frame(self.banner_canvas)
        self.banner_id = self.banner_canvas.create_window((0, 0), window=self.banner_container, anchor='nw', tags=('banner',))
        
        # Create the banner label that will hold the image
        self.banner_label = ttk.Label(self.banner_container)
        self.banner_label.pack(fill=tk.BOTH, expand=True)
        
        # Bind click event to cycle through banners
        self.banner_label.bind("<Button-1>", self.cycle_banner)
        
        # Bind configure event to handle resizing
        self.banner_canvas.bind('<Configure>', self.on_banner_configure)
        
        # Bind mouse wheel for horizontal scrolling
        self.banner_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Store the current banner image
        self.current_banner_img = None
        self.current_banner = 1
        self.update_banner_image()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel for horizontal scrolling"""
        self.banner_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
    
    def on_banner_configure(self, event):
        """Handle banner canvas resizing"""
        try:
            if not hasattr(self, 'banner_canvas') or not self.banner_canvas.winfo_exists():
                return
                
            # Get the current window width
            window_width = self.root.winfo_width()
            
            # Update the canvas size to match the window width
            self.banner_canvas.config(width=window_width)
            
            # Update the scroll region to match the window width
            self.banner_canvas.configure(scrollregion=(0, 0, window_width, self.banner_height))
            
            # Update the banner container size
            if hasattr(self, 'banner_id'):
                self.banner_canvas.itemconfig(self.banner_id, width=window_width, height=self.banner_height)
            
            # Update the banner image to fill the new size
            self.update_banner_image()
            
        except Exception as e:
            self.safe_log(f"Error in on_banner_configure: {str(e)}")
            return
        
    def on_banner_resize(self, event):
        """Handle banner container resize (kept for backward compatibility)"""
        self.on_banner_configure(event)
    
    def safe_log(self, message):
        """Safely log a message, handling cases where log_text might not exist yet"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        print(f"[{timestamp}] {message}")
        
    def on_window_resize(self, event=None):
        """Handle window resize events"""
        if event.widget == self.root:  # Only process main window resize
            self.update_banner_image()
    
    def update_banner_image(self):
        """Update the banner image with proper scaling"""
        if not hasattr(self, 'banner_label') or not hasattr(self, 'banner_canvas') or not self.banner_canvas.winfo_exists():
            return
            
        try:
            # Get the current window width
            window_width = self.root.winfo_width()
            
            # Use the window width for the banner width
            canvas_width = window_width
            canvas_height = self.banner_height
            
            # Ensure minimum dimensions
            min_width = 1200
            if canvas_width < min_width:
                # If window is too small, maintain aspect ratio
                canvas_height = int((min_width / canvas_width) * canvas_height)
                canvas_width = min_width
                
        except (tk.TclError, AttributeError) as e:
            self.safe_log(f"Error getting dimensions: {str(e)}")
            canvas_width = 1200
            canvas_height = 200
            
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            banner_path = os.path.join(script_dir, "assest", f"{self.current_banner}.png")
            
            if not os.path.exists(banner_path):
                self.safe_log(f"Banner image not found: {banner_path}")
                return
                
            # Use the current canvas width for the banner
            container_width = max(100, canvas_width)
            
            # Load the image
            banner_img = Image.open(banner_path)
            
            # Ensure we have valid dimensions
            if banner_img.width == 0 or banner_img.height == 0:
                self.safe_log("Invalid image dimensions")
                return
            
            # Calculate new dimensions while maintaining aspect ratio
            aspect_ratio = banner_img.width / banner_img.height
            new_width = max(100, container_width)  # Ensure minimum width of 100px
            new_height = int(new_width / aspect_ratio)
            
            # Ensure we have valid dimensions before calculations
            if aspect_ratio <= 0:
                self.safe_log(f"Invalid aspect ratio: {aspect_ratio}")
                return
                
            # Calculate new dimensions to fill the available space while maintaining aspect ratio
            try:
                # Always prioritize width for a banner
                new_width = max(1200, canvas_width)  # Minimum width of 1200px
                new_height = int(new_width / aspect_ratio)
                
                # If the calculated height is too small, adjust to maintain minimum height
                if new_height < 150:
                    new_height = 150
                    new_width = int(new_height * aspect_ratio)
                
                # Final validation of dimensions
                if new_width <= 0 or new_height <= 0:
                    self.safe_log(f"Calculated invalid dimensions: {new_width}x{new_height}")
                    return
                    
            except Exception as e:
                self.safe_log(f"Error calculating dimensions: {str(e)}")
                return
            
            # Resize the image with safety check
            try:
                banner_img = banner_img.resize((new_width, new_height), Image.LANCZOS)
            except Exception as e:
                self.safe_log(f"Error resizing image: {str(e)}")
                return
            
            # Convert to PhotoImage and update the label
            try:
                self.current_banner_img = ImageTk.PhotoImage(banner_img)
                self.banner_label.config(image=self.current_banner_img)
                
                # Update the canvas scroll region if it exists
                if hasattr(self, 'banner_canvas') and self.banner_canvas.winfo_exists():
                    self.banner_canvas.configure(scrollregion=self.banner_canvas.bbox("all"))
                    
            except Exception as e:
                self.safe_log(f"Error updating banner display: {str(e)}")
                
        except Exception as e:
            self.safe_log(f"Error updating banner: {str(e)}")
    
    def cycle_banner(self, event=None):
        """Cycle through available banner images"""
        self.current_banner = (self.current_banner % 3) + 1  # We have 3 banners (1.png, 2.png, 3.png)
        self.update_banner_image()
            
    def start_slideshow(self):
        """Start the automatic slideshow of banner images"""
        self.slideshow_running = True
        self.cycle_slideshow()
        
    def stop_slideshow(self):
        """Stop the automatic slideshow"""
        self.slideshow_running = False
        if hasattr(self, 'slideshow_id'):
            self.root.after_cancel(self.slideshow_id)
    
    def cycle_slideshow(self):
        """Cycle to the next banner and schedule the next cycle"""
        if self.slideshow_running:
            self.cycle_banner()
            self.slideshow_id = self.root.after(self.slideshow_interval, self.cycle_slideshow)
    
    def on_close(self):
        """Handle window close event"""
        self.stop_slideshow()
        self.root.destroy()
                    
    def setup_ui(self):
        # Create a main container that will hold everything below the banner
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        
        # Main content frame (everything except banner)
        main_frame = ttk.Frame(self.main_container, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
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
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.update_idletasks()
        print(f"[{timestamp}] {message}")  # Always print to console
        
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
            
            # Log model and scaler information
            if hasattr(self.model, 'n_features_in_'):
                self.log(f"Model expects {self.model.n_features_in_} features")
            
            if self.scaler is None and os.path.exists('saved_models/scaler.joblib'):
                self.scaler = joblib.load('saved_models/scaler.joblib')
                self.log("Loaded default scaler")
                
            if self.label_encoder is None and os.path.exists('saved_models/label_encoder.pkl'):
                with open('saved_models/label_encoder.pkl', 'rb') as f:
                    self.label_encoder = joblib.load(f)
                self.log("Loaded default label encoder")
            
            if self.scaler is None:
                self.log("Warning: No scaler found. Predictions may be inaccurate.")
            else:
                self.log(f"Scaler expects {self.scaler.n_features_in_} features")
                
            if self.label_encoder is None:
                self.log("Warning: No label encoder found. Class labels may be incorrect.")
            
            self.log("Model loaded successfully!")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            self.log(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def _show_prediction_visualization(self, class_names, probabilities, predicted_class):
        """Display a bar chart of prediction probabilities."""
        # Create a new window for the visualization
        vis_window = Toplevel(self.root)
        vis_window.title("Prediction Results")
        vis_window.geometry("800x600")
        
        # Create a frame for the visualization
        vis_frame = ttk.Frame(vis_window)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a figure for the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort the genres and probabilities for better visualization
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = [probabilities[i] * 100 for i in sorted_indices]
        sorted_labels = [class_names[i] for i in sorted_indices]
        
        # Create color list - highlight the predicted class
        colors = ['#3498db' if label != predicted_class else '#e74c3c' for label in sorted_labels]
        
        # Create the bar chart
        bars = ax.bar(sorted_labels, sorted_probs, color=colors)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Customize the plot
        ax.set_title('Genre Prediction Probabilities', fontsize=14, fontweight='bold')
        ax.set_xlabel('Genres', fontsize=12)
        ax.set_ylabel('Confidence (%)', fontsize=12)
        ax.set_ylim(0, 110)  # Leave some space for the value labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add a text box with the prediction result
        prediction_text = f"Predicted Genre: {predicted_class}"
        props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9)
        ax.text(0.02, 0.98, prediction_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # Create a canvas and add it to the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=vis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a close button
        close_btn = ttk.Button(vis_frame, text="Close", command=vis_window.destroy)
        close_btn.pack(pady=10)
        
        # Make the window modal
        vis_window.transient(self.root)
        vis_window.grab_set()
        self.root.wait_window(vis_window)
    
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
            if self.scaler is not None and hasattr(self.scaler, 'transform'):
                try:
                    self.log(f"Scaling features. Scaler expects {self.scaler.n_features_in_} features, got {features.shape[1]}")
                    
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
                    self.log("Features scaled successfully")
                except Exception as e:
                    self.log(f"Warning: Error during feature scaling: {str(e)}")
                    # Continue without scaling if there's an error
            
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
                    
                    # Show visualization if we have probabilities
                    if has_probabilities:
                        self._show_prediction_visualization(class_names, probabilities, predicted_class)
                    
                    # Also show text results in the log
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
                    
                    self.log(f"Prediction complete: {predicted_class}")
                    self.log(result_text)
                    
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
                    precision_score, recall_score, f1_score, silhouette_score
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
                
                # Calculate Silhouette Score (on scaled features)
                try:
                    # Get unique classes to ensure we have more than one class
                    unique_classes = np.unique(y_test)
                    if len(unique_classes) > 1:  # Silhouette score requires at least 2 classes
                        silhouette_avg = silhouette_score(X_test_scaled, y_pred)
                        self._create_metric_frame(notebook, "Silhouette Score", {"Trained Model": silhouette_avg})
                    else:
                        self.log("Silhouette Score not calculated: Only one class present in test set")
                except Exception as e:
                    self.log(f"Could not calculate Silhouette Score: {str(e)}")
                
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
    - Silhouette Score: {silhouette_avg if 'silhouette_avg' in locals() else 'N/A'}

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
        
    def _export_classification_report(self, model_name, y_true, y_pred, y_pred_proba, class_names, output_dir='reports'):
        """Export classification report and metrics to a file."""
        try:
            from datetime import datetime
            import os
            import numpy as np
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=class_names)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate additional metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            # Create report content
            report_content = f"""Classification Report - {model_name}
{'='*80}

Model: {model_name}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Number of Samples: {len(y_true)}
Number of Classes: {len(class_names)}

{'='*80}
Classification Report:
{report}

{'='*80}
Confusion Matrix:
{cm}

{'='*80}
Additional Metrics:
Accuracy: {accuracy:.4f}
"""
            # Save report to file
            report_file = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_report_{timestamp}.txt")
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.log(f"Report saved to {os.path.abspath(report_file)}")
            return True
            
        except Exception as e:
            self.log(f"Error exporting classification report: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _create_roc_curve(self, parent_frame, y_test_encoded, y_pred_proba, model_name=None):
        """Create ROC curve visualization."""
        try:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            from itertools import cycle
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create a new tab for ROC curve
            roc_frame = ttk.Frame(parent_frame)
            parent_frame.add(roc_frame, text="ROC Curve")
            
            # Create a figure for the ROC curve
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            # Binarize the output for multi-class ROC
            n_classes = len(self.label_encoder.classes_)
            y_test_bin = label_binarize(y_test_encoded, classes=range(n_classes))
            
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
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            
            # Adjust legend to be outside the plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
            
            # Add the plot to the tkinter window
            canvas = FigureCanvasTkAgg(fig, master=roc_frame)
            canvas.draw()
            
            # Add a toolbar for navigation
            toolbar = NavigationToolbar2Tk(canvas, roc_frame)
            toolbar.update()
            
            # Pack the canvas and toolbar
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
            
        except Exception as e:
            self.log(f"Error creating ROC curve: {str(e)}")
            messagebox.showerror("Error", f"Failed to create ROC curve: {str(e)}")


class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Set window size and position it in the center of the screen
        window_width = 600  # Increased width to accommodate the layout
        window_height = 450  # Increased height to fit all elements
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Set window background and style
        self.root.configure(bg='#2c3e50')
        self.root.resizable(False, False)  # Prevent window resizing
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TProgressbar', 
                       thickness=20,
                       troughcolor='#34495e',
                       background='#3498db',
                       lightcolor='#3498db',
                       darkcolor='#3498db',
                       bordercolor='#2c3e50',
                       troughrelief='flat',
                       borderwidth=0)
        
        # Load and display the logo
        try:
            logo_path = os.path.join('assest', 'logo.png')
            logo_img = Image.open(logo_path)
            # Resize logo while maintaining aspect ratio
            logo_img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(self.root, image=self.logo, bg='#2c3e50')
            logo_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading logo: {e}")
        
        # Create a main container for better organization
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Top section for logo
        logo_frame = tk.Frame(main_container, bg='#2c3e50')
        logo_frame.pack(fill='x', pady=(0, 10))
        
        # Application name - larger and more prominent
        title_frame = tk.Frame(main_container, bg='#2c3e50')
        title_frame.pack(fill='x', pady=(10, 0))
        
        title_label = tk.Label(
            title_frame, 
            text="GENTECTIVE", 
            font=('Arial Black', 42, 'bold'), 
            fg='#3498db',
            bg='#2c3e50',
            pady=0
        )
        title_label.pack()
        
        # Tagline
        tagline_frame = tk.Frame(main_container, bg='#2c3e50')
        tagline_frame.pack(fill='x', pady=(0, 10))
        
        tagline_label = tk.Label(
            tagline_frame,
            text="Genre Prediction System for Your Tracks",
            font=('Arial', 16, 'italic'),
            fg='#ecf0f1',
            bg='#2c3e50',
            pady=5
        )
        tagline_label.pack()
        
        # Team name section - more prominent
        team_frame = tk.Frame(main_container, bg='#2c3e50')
        team_frame.pack(fill='x', pady=(10, 5))
        
        team_label = tk.Label(
            team_frame,
            text="Developed by ASUS2 Team",
            font=('Arial', 14, 'bold'),
            fg='#bdc3c7',
            bg='#2c3e50',
            pady=5
        )
        team_label.pack()
        
        # Add a decorative line with gradient effect
        line_frame = tk.Frame(main_container, bg='#2c3e50', height=2)
        line_frame.pack(fill='x', pady=10)
        
        canvas = tk.Canvas(line_frame, height=2, bg='#2c3e50', highlightthickness=0)
        canvas.pack(fill='x', padx=20)
        
        # Create gradient line
        width = 600
        for i in range(0, width, 2):
            # Calculate color gradient (darker blue to light blue and back)
            pos = (i / width) * 2
            if pos > 1:
                pos = 2 - pos
            r = int(52 * (1 - pos) + 52 * pos)
            g = int(152 * (1 - pos) + 211 * pos)
            b = int(219 * (1 - pos) + 235 * pos)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(i, 1, i+2, 1, fill=color, width=2)
        
        canvas.config(width=width)
        
        # Loading section
        loading_frame = tk.Frame(main_container, bg='#2c3e50')
        loading_frame.pack(fill='x', pady=(10, 20))
        
        # Loading text with version info
        version_text = tk.Label(
            loading_frame,
            text="Version 1.0.0",
            font=('Arial', 9),
            fg='#95a5a6',
            bg='#2c3e50'
        )
        version_text.pack(pady=(0, 15))
        
        # Loading status with animated dots
        self.loading_text = tk.StringVar()
        self.loading_text.set("Analyzing audio features")
        self.dots = 0
        
        loading_text = tk.Label(
            loading_frame,
            textvariable=self.loading_text,
            font=('Arial', 10, 'bold'),
            fg='#3498db',
            bg='#2c3e50'
        )
        loading_text.pack()
        
        # Copyright text at the bottom
        copyright_frame = tk.Frame(main_container, bg='#2c3e50')
        copyright_frame.pack(side='bottom', pady=10)
        
        copyright_text = tk.Label(
            copyright_frame,
            text="© 2025 ASUS2 Team. All Rights Reserved.",
            font=('Arial', 8),
            fg='#7f8c8d',
            bg='#2c3e50'
        )
        copyright_text.pack()
        
        # Add a styled progress bar with better visibility
        progress_frame = tk.Frame(main_container, bg='#2c3e50')
        progress_frame.pack(fill='x', pady=(20, 10), padx=40)
        
        # Configure progress bar style
        style = ttk.Style()
        style.configure('Custom.Horizontal.TProgressbar',
            thickness=15,
            troughcolor='#34495e',
            background='#3498db',
            troughrelief='flat',
            borderwidth=0,
            lightcolor='#3498db',
            darkcolor='#3498db'
        )
        
        self.progress = ttk.Progressbar(
            progress_frame,
            orient='horizontal',
            length=500,
            mode='determinate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress.pack(expand=True, fill='x')
        
        # Make the window appear on top
        self.root.lift()
        self.root.attributes('-topmost', True)
        
        # Start the progress bar animation
        self.update_progress()
    
    def update_progress(self, value=0):
        # Update loading text with animated dots
        self.dots = (self.dots + 1) % 4
        dots = '.' * self.dots
        self.loading_text.set(f"Analyzing audio features{dots}")
        
        if value <= 100:
            self.progress['value'] = value
            self.root.after(30, self.update_progress, value + 1)
        else:
            self.root.destroy()

def main():
    # Create and show splash screen
    splash_root = tk.Tk()
    splash = SplashScreen(splash_root)
    
    # Simulate loading time
    splash_root.after(3500, splash_root.destroy)
    splash_root.mainloop()
    
    # Create and show main application
    root = tk.Tk()
    
    # Set application icon
    try:
        icon_path = os.path.join('assest', 'logo.png')
        if os.path.exists(icon_path):
            img = Image.open(icon_path)
            photo = ImageTk.PhotoImage(img)
            root.iconphoto(False, photo)
    except Exception as e:
        print(f"Error setting window icon: {e}")
    
    app = GentectiveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()