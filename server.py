# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

# Import custom modules
from preprocessing.crop_face import detect_and_crop_face
from preprocessing.extract_regions import detect_and_extract_regions
from main.redness import calculate_redness

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ✅ Set up paths
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "data/processed_faces"
REGIONS_DIR = "data/extracted_regions"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REGIONS_DIR, exist_ok=True)

def normalize_redness(score, min_value=25, max_value=100):
    """Scale redness scores to a 25-100 range."""
    scaled_score = ((score - 2) / (5 - 2)) * (max_value - min_value) + min_value
    return round(max(min(scaled_score, max_value), min_value), 2)

@app.route('/')
def home():
    """Base route."""
    return jsonify({"message": "✅ Redness detection API is live!"}), 200

@app.route('/healthz')
def health():
    """Health check for Render."""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze redness from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filename)

    # ✅ Step 1: Detect and crop face
    processed_path = os.path.join(PROCESSED_DIR, file.filename)
    face_image = detect_and_crop_face(filename, processed_path)
    if face_image is None:
        return jsonify({"error": "No face detected"}), 400

    # ✅ Step 2: Extract facial regions
    detect_and_extract_regions(processed_path, REGIONS_DIR)

    # ✅ Step 3: Compute redness for each region
    region_scores = {}
    for region in ["forehead", "left_cheek", "right_cheek", "chin"]:
        region_path = os.path.join(REGIONS_DIR, f"{os.path.splitext(file.filename)[0]}_{region}.jpg")
        if os.path.exists(region_path):
            region_image = cv2.imread(region_path)
            raw_score = calculate_redness(region_image)
            region_scores[region] = normalize_redness(raw_score)

    return jsonify({"redness_scores": region_scores})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use Render's PORT if available
    app.run(host='0.0.0.0', port=port)
