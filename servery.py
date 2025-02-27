from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def normalize_brightness(image):
    """Normalize brightness across different lighting conditions."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  # Normalize brightness
    normalized = cv2.merge((l, a, b))
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

def calculate_redness(image):
    """Compute redness score based on HSV color space."""
    image = normalize_brightness(image)  # Normalize brightness before processing
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjusted red detection range
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    redness_mask = mask1 + mask2

    total_pixels = image.shape[0] * image.shape[1]
    red_pixels = np.count_nonzero(redness_mask)
    redness_percentage = (red_pixels / total_pixels) * 100

    return round(redness_percentage, 2)

@app.route('/')
def home():
    """Render's health check endpoint."""
    return jsonify({"message": "✅ Redness detection API is running!"}), 200

@app.route('/healthz')
def health():
    """Health check for Render deployments."""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze redness from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    redness_score = calculate_redness(image)
    return jsonify({"redness_score": redness_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
