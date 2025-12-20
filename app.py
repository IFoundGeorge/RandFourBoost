import os
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import tempfile
import shutil
from datetime import datetime
from algorithm import extract_features, MyCustomAlgorithm
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and encoders
model = None
scaler = None
label_encoder = None

def load_model():
    global model, scaler, label_encoder
    try:
        # Try to load existing model and encoders
        model = joblib.load('models/gentective_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        print("Model and encoders loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Initialize new ones if loading fails
        model = MyCustomAlgorithm()
        scaler = StandardScaler()
        label_encoder = LabelEncoder()
        print("Initialized new model and encoders")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_dir = tempfile.mkdtemp()
            filepath = os.path.join(temp_dir, filename)
            file.save(filepath)
            
            # Extract features
            features = extract_features(filepath)
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            probabilities = model.predict_proba(features_scaled)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_genre = label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Prepare results
            results = {
                'genre': predicted_genre,
                'probabilities': {}
            }
            
            # Add probabilities for all genres
            for i, prob in enumerate(probabilities):
                genre = label_encoder.inverse_transform([i])[0]
                results['probabilities'][genre] = float(prob)
            
            # Generate visualization
            plot_path = generate_visualization(probabilities, label_encoder.classes_)
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return jsonify({
                'success': True,
                'prediction': results,
                'plot_path': plot_path
            })
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

def generate_visualization(probabilities, classes):
    # Create a bar chart of genre probabilities
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(classes))
    
    # Sort by probability
    sorted_idx = np.argsort(probabilities)
    sorted_probs = [probabilities[i] for i in sorted_idx]
    sorted_classes = [classes[i] for i in sorted_idx]
    
    # Plot
    plt.barh(y_pos, sorted_probs, align='center', alpha=0.7)
    plt.yticks(y_pos, sorted_classes)
    plt.xlabel('Probability')
    plt.title('Genre Prediction Probabilities')
    
    # Save to bytes
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert to base64 for embedding in HTML
    plot_data = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{plot_data}"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Load model when starting the app
    load_model()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=True)
