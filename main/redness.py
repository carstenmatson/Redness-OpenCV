# redness.py
import os
import cv2
import numpy as np

def normalize_brightness(image):
    """Normalize brightness using LAB color space and CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def adjust_gamma(image, gamma=1.1):
    """Apply gamma correction for brightness balancing."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def calculate_redness(image):
    """Compute redness score with lighting correction and scaling (25-100)."""
    
    # ✅ Normalize brightness and adjust gamma
    image = normalize_brightness(image)
    image = adjust_gamma(image, gamma=1.1)

    # ✅ Convert to HSV and define red color ranges
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 40, 50]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([160, 40, 50]), np.array([180, 255, 255])

    # ✅ Create redness mask and apply noise reduction
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    redness_mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), np.uint8)
    redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_CLOSE, kernel)

    # ✅ Compute redness coverage & intensity
    total_pixels = image.shape[0] * image.shape[1]
    red_pixels = np.count_nonzero(redness_mask)
    if red_pixels == 0:
        return 25  # Set minimum redness to 25

    redness_intensity = np.mean(hsv[:, :, 1][redness_mask > 0])
    redness_percentage = (red_pixels / total_pixels) * 100
    redness_score = ((redness_percentage * redness_intensity / 255) ** 1.2) * 20  

    # ✅ Ensure redness score stays within 25-100
    return round(max(min(redness_score, 100), 25), 2)

    





