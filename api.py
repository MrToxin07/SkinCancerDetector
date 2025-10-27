import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 100, 75
MODEL_PATH = 'models/skin_cancer_detector.h5'

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run train_model.py first to train and save the model.")
    exit()

# Get the class labels from the training script
class_labels = ['Actinic keratoses and intraepithelial carcinomae',
                'Basal cell carcinoma',
                'Benign keratosis-like lesions',
                'Dermatofibroma',
                'Melanocytic nevi',
                'Melanoma',
                'Vascular lesions']

# Initialize the Flask application
app = Flask(__name__)

# --- Web Page Route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image file from the request
            img_data = file.read()
            img = image.load_img(io.BytesIO(img_data), target_size=(IMG_WIDTH, IMG_HEIGHT))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make the prediction
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class_index = np.argmax(score)
            predicted_class_name = class_labels[predicted_class_index]
            confidence = float(np.max(score) * 100)

            response = {
                "prediction": predicted_class_name,
                "confidence": round(confidence, 2)
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)