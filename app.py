import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 100, 75
MODEL_PATH = 'models/skin_cancer_detector.h5'
COMPARISON_IMAGES_FOLDER = 'static/comparison_images'

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run train_model.py first to train and save the model.")
    exit()

# Get the class labels from the training script and map them to filenames
class_labels = {
    'Melanocytic nevi': 'Melanocytic_nevi',
    'Melanoma': 'Melanoma',
    'Benign keratosis-like lesions': 'Benign_keratosis-like_lesions',
    'Basal cell carcinoma': 'Basal_cell_carcinoma',
    'Actinic keratoses and intraepithelial carcinomae': 'Actinic_keratoses_and_intraepithelial_carcinomae',
    'Vascular lesions': 'Vascular_lesions',
    'Dermatofibroma': 'Dermatofibroma'
}
sorted_labels = sorted(list(class_labels.keys()))

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

            predicted_class_name = sorted_labels[predicted_class_index]
            predicted_class_filename = class_labels[predicted_class_name] + '.jpg'

            confidence = float(np.max(score) * 100)

            response = {
                "prediction": predicted_class_name,
                "confidence": round(confidence, 2),
                "comparison_image": predicted_class_filename
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

# --- Static file serving route for comparison images ---
@app.route('/static/comparison_images/<filename>')
def serve_comparison_image(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'static', 'comparison_images'), filename)

if __name__ == '__main__':
    # This host setting allows anyone on your local network to access the app
    app.run(debug=True, host='0.0.0.0')