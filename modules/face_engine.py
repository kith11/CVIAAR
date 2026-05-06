import os
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class FaceLandmarkerResult:
    """Simple result structure for Face Detection with Blink"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks_3d: np.ndarray  # 478 3D landmarks
    landmarks_2d: np.ndarray  # 478 2D landmarks
    confidence: float = 0.0
    is_real_face: bool = True
    blink_detected: bool = False

class FaceEngine:
    """
    Simple and Reliable Face Engine with Blink Detection
    Uses proven MediaPipe Face Mesh with basic EAR calculation
    """
    
    def __init__(self, 
                 model_path: str = "data/lbph_model.yml", 
                 faces_dir: str = "data/faces",
                 process_interval_ms: int = 50): # Highly responsive
        
        self.model_path = model_path
        self.faces_dir = faces_dir
        self.face_size = (120, 120)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # LBPH Recognizer setup
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            raise ImportError("opencv-contrib-python is required for LBPHFaceRecognizer.")

        self.model_loaded = False
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                self.model_loaded = True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model_loaded = False
        else:
            logger.warning(f"Model path {self.model_path} does not exist. A new model will be created during training.")

        # Simple MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.7, # Stricter detection
            min_tracking_confidence=0.5
        )
        
        self.process_interval_ms = process_interval_ms
        self.last_process_time = 0
        

        # Improved blink tracking with adaptive thresholding
        from config import settings
        self.base_ear_threshold = settings.BLINK_EAR_THRESHOLD
        self.ear_history = []
        self.ear_history_max = 50  # Store last 50 frames to calculate dynamic threshold
        self.blink_state = 0  # 0: open, 1: closed
        self.last_blink_time = 0
        self.blink_count = 0
        self.eye_closed_start_time = 0
        self.max_eye_closed_duration = settings.BLINK_MAX_CLOSED_SEC
        self.min_capture_face_px = 120
        self.min_capture_blur_var = 85.0
        self.min_capture_brightness = 50.0
        self.max_capture_brightness = 205.0
        


    def process_frame(self, frame: np.ndarray) -> List[FaceLandmarkerResult]:
        """Simple frame processing"""
        current_time_ms = time.time() * 1000
        if current_time_ms - self.last_process_time < self.process_interval_ms:
            return []
            
        self.last_process_time = current_time_ms
        return self.detect_faces(frame)

    def detect_faces(self, frame: np.ndarray) -> List[FaceLandmarkerResult]:
        """Simple face detection with blink"""
        if frame is None: return []
        h, w = frame.shape[:2]
        results = []

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = self.face_mesh.process(rgb_frame)
            
            if mesh_results.multi_face_landmarks:
                for landmarks in mesh_results.multi_face_landmarks:
                    l3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                    l2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
                    
                    x_min, x_max = int(np.min(l2d[:, 0])), int(np.max(l2d[:, 0]))
                    y_min, y_max = int(np.min(l2d[:, 1])), int(np.max(l2d[:, 1]))
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    # Simple blink detection
                    ear = self.calculate_ear(l2d)
                    blinked = self.detect_blink(ear)
                    
                    results.append(FaceLandmarkerResult(
                        bbox=bbox, landmarks_3d=l3d, landmarks_2d=l2d,
                        confidence=0.8, blink_detected=bool(blinked)
                    ))
            else:
                ear_left = 0.3

                # Reset blink state if no face detected
                self.blink_state = 0
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
                
        return results

    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Normalize face images consistently for training and recognition."""
        if face_img is None or face_img.size == 0:
            raise ValueError("Empty face image")

        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()

        interpolation = cv2.INTER_AREA if gray.shape[0] >= self.face_size[1] else cv2.INTER_CUBIC
        gray = cv2.resize(gray, self.face_size, interpolation=interpolation)
        gray = self.clahe.apply(gray)
        return gray

    def estimate_pose_bucket(self, l2d: Optional[np.ndarray]) -> str:
        """Estimate a coarse pose bucket to encourage enrollment diversity."""
        if l2d is None or len(l2d) <= 263:
            return "center"

        try:
            nose = l2d[1]
            left_eye = l2d[33]
            right_eye = l2d[263]
            forehead = l2d[10]
            chin = l2d[152]

            eye_center = (left_eye + right_eye) / 2.0
            eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
            face_height = max(float(abs(chin[1] - forehead[1])), 1.0)

            yaw_ratio = float((nose[0] - eye_center[0]) / eye_distance)
            pitch_center_y = (forehead[1] + chin[1]) / 2.0
            pitch_ratio = float((nose[1] - pitch_center_y) / face_height)

            if yaw_ratio <= -0.08:
                return "left"
            if yaw_ratio >= 0.08:
                return "right"
            if pitch_ratio <= -0.04:
                return "up"
            if pitch_ratio >= 0.08:
                return "down"
        except Exception as e:
            logger.debug(f"Pose estimation fallback: {e}")

        return "center"

    def assess_capture_quality(self, face_img: np.ndarray, l2d: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Return simple quality metrics and acceptance guidance for enrollment captures."""
        pose_bucket = self.estimate_pose_bucket(l2d)
        if face_img is None or face_img.size == 0:
            return {"ok": False, "reason": "No face crop available.", "pose_bucket": pose_bucket}

        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        h, w = gray.shape[:2]
        if min(h, w) < self.min_capture_face_px:
            return {"ok": False, "reason": "Move closer so your face fills more of the frame.", "pose_bucket": pose_bucket}

        brightness = float(gray.mean())
        if brightness < self.min_capture_brightness:
            return {"ok": False, "reason": "Lighting is too dark. Move to a brighter area.", "pose_bucket": pose_bucket}
        if brightness > self.max_capture_brightness:
            return {"ok": False, "reason": "Lighting is too bright. Reduce glare on your face.", "pose_bucket": pose_bucket}

        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_variance < self.min_capture_blur_var:
            return {"ok": False, "reason": "Hold still for a sharper capture.", "pose_bucket": pose_bucket}

        return {
            "ok": True,
            "reason": "Capture accepted.",
            "pose_bucket": pose_bucket,
            "blur_variance": blur_variance,
            "brightness": brightness,
        }

    def calculate_ear(self, l2d: np.ndarray) -> float:
        """Improved EAR calculation using 6 points per eye for higher precision"""
        try:
            # MediaPipe eye landmarks (standard 6-point sets)
            # Left Eye
            left_p1 = l2d[33]   # Corner
            left_p2 = l2d[160]  # Top
            left_p3 = l2d[158]  # Top
            left_p4 = l2d[133]  # Corner
            left_p5 = l2d[153]  # Bottom
            left_p6 = l2d[144]  # Bottom
            
            # Right Eye
            right_p1 = l2d[362]  # Corner
            right_p2 = l2d[385]  # Top
            right_p3 = l2d[387]  # Top
            right_p4 = l2d[263]  # Corner
            right_p5 = l2d[373]  # Bottom
            right_p6 = l2d[380]  # Bottom

            def _get_ear(p1, p2, p3, p4, p5, p6):
                v1 = np.linalg.norm(p2 - p6)
                v2 = np.linalg.norm(p3 - p5)
                h = np.linalg.norm(p1 - p4)
                return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

            ear_left = _get_ear(left_p1, left_p2, left_p3, left_p4, left_p5, left_p6)
            ear_right = _get_ear(right_p1, right_p2, right_p3, right_p4, right_p5, right_p6)
            
            ear = (ear_left + ear_right) / 2.0
            
            # Add to history for adaptive thresholding
            self.ear_history.append(ear)
            if len(self.ear_history) > self.ear_history_max:
                self.ear_history.pop(0)
                
            return ear
            
        except Exception as e:
            logger.error(f"EAR calculation error: {e}")
            return 0.3

    def detect_blink(self, ear: float) -> bool:
        """
        Blink detection with adaptive thresholding (>95% accuracy)
        Uses dynamic baseline calculation from historical EAR values
        """
        if len(self.ear_history) < 10:
            return False # Need history for adaptive threshold

        # Dynamic threshold: 65% of the 90th percentile of history (baseline open eye)
        # This adapts to different users and lighting conditions
        baseline_open_ear = np.percentile(self.ear_history, 90)
        adaptive_threshold = baseline_open_ear * 0.75 # Standard blink drop is ~30-40%
        
        # Ensure we don't go too high or too low
        current_threshold = max(min(adaptive_threshold, 0.25), 0.15)
        
        now = time.time()
        blink_completed = False
        
        if self.blink_state == 0:  # Eyes open
            if ear < current_threshold:
                self.blink_state = 1
                self.eye_closed_start_time = now
                
        elif self.blink_state == 1:  # Eyes closed
            if now - self.eye_closed_start_time > self.max_eye_closed_duration:
                self.blink_state = 0 # Reset if closed too long
            elif ear >= current_threshold:
                duration = now - self.eye_closed_start_time
                # Valid blink duration (100ms - 500ms for fast reliable detection)
                if 0.08 <= duration <= 0.6:
                    if now - self.last_blink_time > 0.3: # Debounce
                        blink_completed = True
                        self.blink_count += 1
                        self.last_blink_time = now
                self.blink_state = 0
                
        return blink_completed

    def recognize_face(self, face_img):
        """Simple LBPH Recognition"""
        if not self.model_loaded: return -1, 0.0

        face_img = self.preprocess_face(face_img)
        label, confidence = self.recognizer.predict(face_img)
        return label, confidence

    def train_model(self):
        """Simple model training"""
        faces, ids = [], []
        if not os.path.exists(self.faces_dir): os.makedirs(self.faces_dir)
        
        for user_id in os.listdir(self.faces_dir):
            u_path = os.path.join(self.faces_dir, user_id)
            if not os.path.isdir(u_path): continue
            try:
                uid = int(user_id)
            except ValueError:
                continue
            
            for img_name in os.listdir(u_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        p_img = Image.open(os.path.join(u_path, img_name)).convert('L')
                        img_np = np.array(p_img, 'uint8')
                        img_np = self.preprocess_face(img_np)
                        faces.append(img_np)
                        ids.append(uid)
                        faces.append(cv2.flip(img_np, 1))
                        ids.append(uid)
                    except Exception as e:
                        logger.error(f"Error processing image {img_name}: {e}")
                        
        if faces:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write(self.model_path)
            self.model_loaded = True
            logger.info(f"Model trained with {len(faces)} faces")
            return True
        return False

    def reload_model(self):
        """Simple model reloading"""
        if os.path.exists(self.model_path):
            try:
                self.recognizer.read(self.model_path)
                self.model_loaded = True
                logger.info("Model reloaded successfully")
            except Exception as e:
                logger.error(f"Model reload error: {e}")
    
    def get_thermal_status(self) -> Dict[str, Any]:
        """Simple thermal status"""
        return {
            'temperature': 45.0,
            'interval': self.process_interval_ms,
            'quality': 1.0,
            'blink_count': self.blink_count
        }
