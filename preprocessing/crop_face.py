import cv2
import os
import mediapipe as mp
from datetime import datetime

# ✅ Define Paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_IMAGES_DIR = os.path.join(root_dir, "data/raw_images")
PROCESSED_IMAGES_DIR = os.path.join(root_dir, "data/processed_faces")
IMAGE_RESIZE_WIDTH = 640  # ✅ Adjust width for consistent processing

# ✅ Ensure Required Directories Exist
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# ✅ Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def resize_image(image, width):
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(width * aspect_ratio)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)


def detect_and_crop_faces(image_path, output_dir):
    """Detect faces and save cropped images."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ERROR: Cannot read image: {image_path}")
        return

    # ✅ Resize for consistent processing
    resized_image = resize_image(image, IMAGE_RESIZE_WIDTH)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # ✅ Detect faces
    results = face_detection.process(rgb_image)

    if results.detections:
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = resized_image.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            # ✅ Adjust bounding box (increase area)
            top_margin = int(0.03 * h)
            bottom_margin = int(0.02 * h)
            x, y, w, h = max(0, x), max(0, y - top_margin), min(w, iw - x), min(h + top_margin + bottom_margin, ih - y + top_margin)

            # ✅ Save Cropped Face
            cropped_face = resized_image[y:y+h, x:x+w]
            unique_suffix = datetime.now().strftime("%Y%m%d%H%M%S%f")
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_face{i}_{unique_suffix}.jpg")
            cv2.imwrite(output_path, cropped_face)
            print(f"✅ Saved cropped face: {output_path}")
    else:
        print(f"❌ No face detected in: {image_path}")


def process_all_images(input_dir, output_dir):
    """Process all images in the input directory."""
    if not os.path.exists(input_dir):
        print(f"❌ ERROR: Input directory does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        detect_and_crop_faces(image_path, output_dir)


if __name__ == "__main__":
    process_all_images(RAW_IMAGES_DIR, PROCESSED_IMAGES_DIR)


