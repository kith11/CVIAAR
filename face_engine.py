import cv2
import mediapipe as mp
import os
import numpy as np
from PIL import Image

class FaceEngine:
    def __init__(self, model_path='data/lbph_model.yml', faces_dir='data/faces'):
        self.model_path = model_path
        self.faces_dir = faces_dir
        
        # Initialize MediaPipe Face Mesh (for liveness)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize LBPH Recognizer
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            raise ImportError("opencv-contrib-python is required for LBPHFaceRecognizer. Please install it.")
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self.model_loaded = True
        else:
            self.model_loaded = False

    def detect_faces_mesh(self, frame):
        """
        Detect faces using Face Mesh.
        Returns list of tuples: (bbox, landmarks)
        bbox: (x, y, w, h)
        landmarks: list of (x, y) normalized coordinates
        """
        # Resize frame for faster processing
        h, w, _ = frame.shape
        scale_factor = 0.5  # Process at half resolution (320x240 for 640x480)
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        faces_data = []
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Calculate bounding box from landmarks
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                # Convert landmarks to list and scale back up
                landmark_points = []
                for lm in landmarks.landmark:
                    # Use original w, h for coordinates
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmark_points.append((x, y))
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                # Add padding
                pad_w = int((x_max - x_min) * 0.2)
                pad_h = int((y_max - y_min) * 0.2)
                
                x_min = max(0, x_min - pad_w)
                y_min = max(0, y_min - pad_h)
                x_max = min(w, x_max + pad_w)
                y_max = min(h, y_max + pad_h)
                
                bw = x_max - x_min
                bh = y_max - y_min
                
                faces_data.append(((x_min, y_min, bw, bh), landmarks.landmark))
                
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
                    
                    # We assume the images stored are already cropped faces, 
                    # but if not, we might need detection here too.
                    # The requirement says "Capture 10 images per user from webcam",
                    # usually these are cropped. I'll assume they are cropped or full frames.
                    # To be safe, let's detect faces in training images if they are full frames.
                    # However, for simplicity and speed on Pi, it's better if we save cropped faces during capture.
                    # I will ensure capture saves cropped faces.
                    
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
            # LBPH returns confidence as distance (lower is better)
            label, confidence = self.recognizer.predict(face_img)
            return label, confidence
        except Exception as e:
            print(f"Recognition error: {e}")
            return -1, 0.0
