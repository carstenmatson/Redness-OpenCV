import os
import sys
import cv2
import numpy as np

# ✅ Ensure the script runs from the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)  # 🔥 FIX: Add root directory to sys.path

# ✅ Import Modules After sys.path Fix
from preprocessing.crop_face import process_all_images as crop_faces
from preprocessing.extract_regions import process_all_images as extract_regions
from main.redness import process_redness  # 🔥 FIX: Use 'main.redness'

# ✅ Define Paths
RAW_IMAGES_DIR = os.path.join(root_dir, "data", "raw_images")
PROCESSED_IMAGES_DIR = os.path.join(root_dir, "data", "processed_faces")
REGIONS_DIR = os.path.join(root_dir, "data", "extracted_regions")
VISUALIZATION_DIR = os.path.join(root_dir, "visualizations")

# ✅ Ensure directories exist
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(REGIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def analyze_skin():
    """
    Pipeline to analyze skin redness on different facial regions.

    Returns:
        dict: Redness scores for different face regions.
    """
    print(f"📸 Processing all images in: {RAW_IMAGES_DIR}")

    # ✅ Step 1: Detect & Crop Faces
    crop_faces(RAW_IMAGES_DIR, PROCESSED_IMAGES_DIR)

    # ✅ Step 2: Extract Facial Regions
    extract_regions(PROCESSED_IMAGES_DIR, REGIONS_DIR)

    # ✅ Step 3: Compute Redness Scores
    redness_scores = process_redness(REGIONS_DIR)  # 🔥 FIX: Only pass one argument

    print("✅ Redness Analysis Complete:", redness_scores)
    return redness_scores


if __name__ == "__main__":
    # ✅ Run analysis on all images in data/raw_images/
    if os.listdir(RAW_IMAGES_DIR):
        scores = analyze_skin()
        print("\n📊 Final Redness Scores:", scores)
    else:
        print("❌ ERROR: No images found! Please place images in data/raw_images/")


