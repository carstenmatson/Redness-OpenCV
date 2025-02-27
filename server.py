# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

# ✅ Corrected import paths
from preprocessing.crop_face import detect_and_crop_faces
from preprocessing.extract_regions import detect_and_extract_regions
from main.redness import calculate_redness

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# ✅ Set up absolute paths (Fix for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_faces")
REGIONS_DIR = os.path.join(BASE_DIR, "data/extracted_regions")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REGIONS_DIR, exist_ok=True)

def normalize_redness(score, min_value=25, max_value=100):
    """Ensure redness scores stay within 25-100."""
    return round(max(min(score, max_value), min_value), 2)

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
    detect_and_crop_faces(filename, PROCESSED_DIR)

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

    if not region_scores:
        return jsonify({"error": "No facial regions detected"}), 400

    return jsonify({"redness_scores": region_scores})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Use Render's PORT if available
    app.run(host='0.0.0.0', port=port)


