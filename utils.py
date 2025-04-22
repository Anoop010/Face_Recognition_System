import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
MODEL_PATH = "model/face_model.yml"
LABEL_MAP_PATH = "model/label_map.npy"

# Ensure necessary directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def add_person(name, image):
    """ Adds a new person's face to the dataset. """
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    img_path = os.path.join(person_dir, f"{len(os.listdir(person_dir))}.jpg")
    cv2.imwrite(img_path, image)
    return img_path


def remove_person(name):
    """ Removes a person from the dataset. """
    person_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(person_dir):
        for file in os.listdir(person_dir):
            os.remove(os.path.join(person_dir, file))
        os.rmdir(person_dir)


def train_model():
    """ Trains a face recognition model using LBPH. """
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # ✅ Fixed the initialization
    faces, labels = [], []
    label_map = {}

    label_id = 0
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_dir):
            label_map[label_id] = person_name
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"⚠ Skipping corrupt image: {img_path}")
                    continue

                faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces_detected:
                    faces.append(img[y:y + h, x:x + w])
                    labels.append(label_id)

            label_id += 1

    if not faces:
        raise ValueError("No faces detected in dataset! Please add more images.")

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    np.save(LABEL_MAP_PATH, label_map)  # ✅ Ensure the dictionary is saved properly
    print("✅ Model trained successfully.")


def recognize_face(image):
    """ Recognizes a face in an image using the trained model. """
    if not os.path.exists(MODEL_PATH):
        return "No Model Found", 0, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # ✅ Fixed recognizer initialization
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y + h, x:x + w]

        try:
            label, confidence = recognizer.predict(face)
            return label_map.get(label, "Unknown"), confidence, (x, y, w, h)
        except Exception as e:
            print(f"⚠ Error in recognition: {e}")
            return "Recognition Error", 0, None

    return "No face detected", 0, None
