import os
import cv2
import numpy as np

def normalize_brightness(image):
    """
    Normalize brightness across the image to reduce the impact of uneven lighting.
    Uses LAB color space and CLAHE (adaptive histogram equalization).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # ✅ Further Reduced CLAHE Clip Limit to Prevent Overcorrection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 🔥 Lowered from 2.5 → 2.0
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_gamma(image, gamma=1.1):
    """
    Apply gamma correction to balance brightness.
    Helps normalize shadows without over-brightening.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def calculate_redness(image):
    """Compute redness score with balanced lighting correction."""

    # ✅ Step 1: Normalize brightness (Less Aggressive)
    image = normalize_brightness(image)
    image = adjust_gamma(image, gamma=1.1)  # 🔥 Lowered from 1.2 → 1.1 (Smoother)

    # ✅ Step 2: Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ✅ Step 3: Define Adjusted Red Ranges (Optimized for Skin Tones)
    lower_red1, upper_red1 = np.array([0, 40, 50]), np.array([15, 255, 255])  # Adjusted Lower Threshold
    lower_red2, upper_red2 = np.array([160, 40, 50]), np.array([180, 255, 255])  

    # ✅ Step 4: Create Redness Mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    redness_mask = cv2.bitwise_or(mask1, mask2)

    # ✅ Step 5: Apply Morphology to Reduce Noise
    kernel = np.ones((3, 3), np.uint8)  
    redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_CLOSE, kernel)

    # ✅ Step 6: Calculate Red Pixel Coverage & Intensity
    total_pixels = image.shape[0] * image.shape[1]
    red_pixels = np.count_nonzero(redness_mask)

    if red_pixels == 0:
        return 0  # No redness detected

    # ✅ Compute Redness Intensity (Mean Saturation in Red Areas)
    redness_intensity = np.mean(hsv[:, :, 1][redness_mask > 0])

    # ✅ New **Balanced** Redness Score Formula (No More 100 Maxing)
    redness_percentage = (red_pixels / total_pixels) * 100
    redness_score = ((redness_percentage * redness_intensity / 255) ** 1.2) * 20  # 🔥 Lowered Power to 1.2 & Scaling to 20

    return round(min(redness_score, 100), 2)  # Keep values in 0-100 range

def process_redness(region_dir):
    """Process all extracted facial regions and compute redness scores."""
    
    redness_scores = {}
    
    for region_file in os.listdir(region_dir):
        if region_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(region_dir, region_file)
            image = cv2.imread(image_path)

            if image is not None:
                score = calculate_redness(image)
                region_name = os.path.splitext(region_file)[0]
                redness_scores[region_name] = score
                print(f"✅ {region_name}: Redness Score = {score}/100")
            else:
                print(f"❌ Failed to read image: {region_file}")

    return redness_scores





