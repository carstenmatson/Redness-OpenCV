import os
import cv2
import numpy as np
import mediapipe as mp

# ✅ Define Paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_IMAGES_DIR = os.path.join(root_dir, "data/processed_faces")
REGIONS_DIR = os.path.join(root_dir, "data/extracted_regions")

# ✅ Ensure Required Directories Exist
os.makedirs(REGIONS_DIR, exist_ok=True)

# ✅ Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# ✅ Facial Landmark Indices
LANDMARKS = {
    "left_cheek": [162, 156, 226, 31, 228, 229, 230, 231, 232, 233, 245, 236, 209, 49, 48, 235, 60, 2, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 136, 172, 58, 132, 93, 234, 127],
    "right_cheek": [365, 397, 288, 435, 361, 401, 323, 366, 454, 356, 389, 353, 446, 261, 448, 449, 450, 451, 452, 453, 464, 351, 412, 437, 420, 429, 279, 278, 392, 290, 2, 0, 267, 269, 270, 409, 375, 321, 405, 314, 17],
    "forehead": [21, 67, 332, 284, 251, 336, 8],
    "chin": [136, 17, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150],
}


def extract_region(image, landmarks, region_name):
    """Extracts a specific facial region using facial landmarks."""
    region_points = [landmarks[i] for i in LANDMARKS[region_name]]

    # ✅ Create a Mask
    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, [np.array(region_points, dtype=np.int32)], 255)

    # ✅ Apply Mask
    return cv2.bitwise_and(image, image, mask=mask)


def detect_and_extract_regions(image_path, output_dir):
    """Detects face landmarks and extracts facial regions."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ERROR: Cannot read image: {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print(f"❌ No face detected in: {image_path}")
        return

    h, w, _ = image.shape
    landmarks = [(int(p.x * w), int(p.y * h)) for p in results.multi_face_landmarks[0].landmark]

    # ✅ Extract & Save Each Region
    for region_name in LANDMARKS.keys():
        region = extract_region(image, landmarks, region_name)
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{region_name}.jpg")
        cv2.imwrite(output_path, region)
        print(f"✅ Saved {region_name} region: {output_path}")


def process_all_images(input_dir, output_dir):
    """Processes all images to extract facial regions."""
    if not os.path.exists(input_dir):
        print(f"❌ ERROR: Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        detect_and_extract_regions(image_path, output_dir)


if __name__ == "__main__":
    process_all_images(PROCESSED_IMAGES_DIR, REGIONS_DIR)





