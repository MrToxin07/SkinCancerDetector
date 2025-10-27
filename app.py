# -------------------------------------------------------------------------
# --- TFLite and Flask Imports ---
# This is the optimal setup for Render's free tier
# -------------------------------------------------------------------------
from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from tflite_runtime.interpreter import Interpreter
from PIL import Image # Pillow is used for image loading/processing
import numpy as np
import io
import os
import sys

# -------------------------------------------------------------------------
# --- 1. PATH CONFIGURATION (Correct for Docker/Render) ---
# -------------------------------------------------------------------------
# NOTE: In Docker, the current working directory (os.getcwd()) is set to /app.
# We define paths relative to /app.
PROJECT_DIR = os.getcwd() 
TFLITE_MODEL_PATH = os.path.join(PROJECT_DIR, 'models', 'skin_cancer_detector.tflite')

# --- Other Configuration ---
IMG_WIDTH, IMG_HEIGHT = 100, 75

# Get the class labels and map them for prediction output
class_labels = {
    'Melanocytic nevi': 'Melanocytic_nevi', 'Melanoma': 'Melanoma',
    'Benign keratosis-like lesions': 'Benign_keratosis-like_lesions',
    'Basal cell carcinoma': 'Basal_cell_carcinoma',
    'Actinic keratoses and intraepithelial carcinomae': 'Actinic_keratoses_and_intraepithelial_carcinomae',
    'Vascular lesions': 'Vascular_lesions', 'Dermatofibroma': 'Dermatofibroma'
}
# The model's index output matches the sorted order of these keys
sorted_labels = sorted(list(class_labels.keys())) 

# -------------------------------------------------------------------------
# --- 2. LOAD TFLite MODEL (Initialization) ---
# -------------------------------------------------------------------------
try:
    # 1. Initialize the TFLite Interpreter
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")
except Exception as e:
    # If the model fails to load, the web app will crash (Render will show an error)
    print(f"FATAL ERROR: Could not load TFLite model at {TFLITE_MODEL_PATH}")
    print(f"Details: {e}")
    # Render requires the process to exit immediately if startup fails
    sys.exit(f"Application startup failed: {e}") 

# -------------------------------------------------------------------------
# --- 3. FLASK APPLICATION ROUTES ---
# -------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # 1. Read and Process Image using PIL
            img = Image.open(io.BytesIO(file.read()))
            img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
            
            # 2. Convert to NumPy array and normalize
            input_data = np.asarray(img_resized, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            input_data /= 255.0  # Normalize to 0-1 range

            # 3. Run TFLite Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            # 4. Get Scores and Confidence (Softmax in NumPy)
            # This is the pure NumPy replacement for tf.nn.softmax
            exp_scores = np.exp(output[0])
            scores = exp_scores / np.sum(exp_scores)
            
            predicted_class_index = np.argmax(scores)
            predicted_class_name = sorted_labels[predicted_class_index]
            confidence = float(np.max(scores) * 100)
            
            # Map the class name to the physical image file name
            predicted_filename = class_labels[predicted_class_name] + '.jpg'

            response_data = {
                "prediction": predicted_class_name,
                "confidence": round(confidence, 2),
                "comparison_image": predicted_filename
            }
            
            # Add headers to prevent browser caching
            response = make_response(jsonify(response_data))
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            return response

        except Exception as e:
            # Catch internal errors during prediction and return a helpful JSON response
            return jsonify({'error': f"Prediction failed due to: {str(e)}"}), 500

# --- Static file serving route for comparison images ---
# This serves files from the 'static/comparison_images' folder, which Docker copied to /app/static/comparison_images
@app.route('/static/comparison_images/<filename>')
def serve_comparison_image(filename):
    # Render's default static file serving will handle this, but for completeness, 
    # we point to the directory inside the container.
    static_folder_path = os.path.join(PROJECT_DIR, 'static', 'comparison_images')
    return send_from_directory(static_folder_path, filename)

# Render requires the app to listen on the $PORT environment variable, which defaults to 8080.
# The CMD line in the Dockerfile handles starting gunicorn on 0.0.0.0:8080.
# We do not include app.run() here.
