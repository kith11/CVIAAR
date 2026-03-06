import os
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


class FaceEngine:
    def __init__(self, model_path: str = "data/lbph_model.yml", faces_dir: str = "data/faces"):
        self.model_path = model_path
        self.faces_dir = faces_dir

        # MediaPipe Face Mesh (landmarks + liveness)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # LBPH Recognizer
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            raise ImportError(
                "opencv-contrib-python is required for LBPHFaceRecognizer. Please install it."
            )

        self.model_loaded = False
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self.model_loaded = True

        # Tunables
        self.face_scale_factor = 0.5  # downscale for mesh processing
        self.face_padding_ratio = 0.20

    def reload_model(self):
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self.model_loaded = True
            print("Face recognition model reloaded.")
        else:
            self.model_loaded = False
            print("Model file not found, cannot reload.")

    def preprocess_face(self, face_img):
        """
        Standardize face image:
        1. Resize to fixed size (100x100)
        2. Histogram Equalization (Lighting normalization)
        """
        # Resize
        face_img = cv2.resize(face_img, (100, 100))
        # Equalize Histogram
        face_img = cv2.equalizeHist(face_img)
        return face_img

    def detect_faces_mesh(self, frame):
        """
        Detect faces using Face Mesh.
        Returns list of tuples: (bbox, landmarks)
        bbox: (x, y, w, h)
        landmarks: list of (x, y) normalized coordinates
        """
        if frame is None:
            return []

        h, w, _ = frame.shape
        sf = float(self.face_scale_factor)
        if sf <= 0 or sf > 1:
            sf = 1.0

        # Process a downscaled frame for speed, but correctly re-project coords.
        if sf != 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=sf, fy=sf)
        else:
            small_frame = frame

        sh, sw, _ = small_frame.shape
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        faces_data = []
        if not results.multi_face_landmarks:
            return faces_data

        for lmset in results.multi_face_landmarks:
            # Compute bbox in small-frame pixel space then scale to original.
            x_min_s, y_min_s = sw, sh
            x_max_s, y_max_s = 0, 0

            for lm in lmset.landmark:
                x_s = int(lm.x * sw)
                y_s = int(lm.y * sh)
                x_min_s = min(x_min_s, x_s)
                y_min_s = min(y_min_s, y_s)
                x_max_s = max(x_max_s, x_s)
                y_max_s = max(y_max_s, y_s)

            # Scale bbox back up to original frame size
            if sf != 1.0:
                x_min = int(x_min_s / sf)
                y_min = int(y_min_s / sf)
                x_max = int(x_max_s / sf)
                y_max = int(y_max_s / sf)
            else:
                x_min, y_min, x_max, y_max = x_min_s, y_min_s, x_max_s, y_max_s

            # Pad bbox
            pad_w = int((x_max - x_min) * float(self.face_padding_ratio))
            pad_h = int((y_max - y_min) * float(self.face_padding_ratio))
            x_min = max(0, x_min - pad_w)
            y_min = max(0, y_min - pad_h)
            x_max = min(w, x_max + pad_w)
            y_max = min(h, y_max + pad_h)

            bw = max(0, x_max - x_min)
            bh = max(0, y_max - y_min)
            if bw == 0 or bh == 0:
                continue

            faces_data.append(((x_min, y_min, bw, bh), lmset.landmark))

        return faces_data

    def calculate_ear(self, landmarks, indices, w, h):
        """
        Calculate Eye Aspect Ratio for a given eye indices.
        """
        # Vertical lines
        A = np.linalg.norm(np.array([landmarks[indices[1]].x * w, landmarks[indices[1]].y * h]) - 
                           np.array([landmarks[indices[5]].x * w, landmarks[indices[5]].y * h]))
        B = np.linalg.norm(np.array([landmarks[indices[2]].x * w, landmarks[indices[2]].y * h]) - 
                           np.array([landmarks[indices[4]].x * w, landmarks[indices[4]].y * h]))
        # Horizontal line
        C = np.linalg.norm(np.array([landmarks[indices[0]].x * w, landmarks[indices[0]].y * h]) - 
                           np.array([landmarks[indices[3]].x * w, landmarks[indices[3]].y * h]))
        
        ear = (A + B) / (2.0 * C)
        return ear

    def check_liveness(self, landmarks, w, h):
        """
        Check blink using EAR.
        Returns average EAR of both eyes.
        """
        # Left eye indices (33, 160, 158, 133, 153, 144)
        left_indices = [33, 160, 158, 133, 153, 144]
        # Right eye indices (362, 385, 387, 263, 373, 380)
        right_indices = [362, 385, 387, 263, 373, 380]
        
        left_ear = self.calculate_ear(landmarks, left_indices, w, h)
        right_ear = self.calculate_ear(landmarks, right_indices, w, h)
        
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear

    def detect_faces(self, frame):
        # Backward compatibility wrapper if needed, 
        # but we prefer using detect_faces_mesh for liveness
        faces_data = self.detect_faces_mesh(frame)
        return [f[0] for f in faces_data]

    def train_model(self):
        """
        Train the LBPH model using images in data/faces directory.
        """
        faces = []
        ids = []
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

        # Traverse all user directories
        for user_id in os.listdir(self.faces_dir):
            user_path = os.path.join(self.faces_dir, user_id)
            if not os.path.isdir(user_path):
                continue
                
            try:
                uid = int(user_id)
            except ValueError:
                continue

            for image_name in os.listdir(user_path):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(user_path, image_name)
                    # Load image, convert to grayscale
                    pil_image = Image.open(image_path).convert('L')
                    image_np = np.array(pil_image, 'uint8')
                    
                    # Preprocess
                    image_np = self.preprocess_face(image_np)
                    
                    faces.append(image_np)
                    ids.append(uid)
        
        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write(self.model_path)
            self.model_loaded = True
            return True
        return False

    def recognize_face(self, face_img):
        """
        Recognize a face image (grayscale, cropped).
        Returns (label, confidence).
        """
        if not self.model_loaded:
            return -1, 0.0
        
        try:
            # Preprocess
            face_img = self.preprocess_face(face_img)
            
            # LBPH returns confidence as distance (lower is better)
            label, confidence = self.recognizer.predict(face_img)
            return label, confidence
        except Exception as e:
            print(f"Recognition error: {e}")
            return -1, 0.0
