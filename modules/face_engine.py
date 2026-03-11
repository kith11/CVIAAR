import os
import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
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
                 process_interval_ms: int = 100): # Reasonable interval
        
        self.model_path = model_path
        self.faces_dir = faces_dir
        
        # LBPH Recognizer setup
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            raise ImportError("opencv-contrib-python is required for LBPHFaceRecognizer.")

        self.model_loaded = False
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self.model_loaded = True

        # Simple MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        self.process_interval_ms = process_interval_ms
        self.last_process_time = 0
        
        # Improved blink tracking
        from config import settings
        self.ear_threshold = settings.BLINK_EAR_THRESHOLD  # Configurable threshold
        self.blink_state = 0  # 0: open, 1: closed
        self.last_blink_time = 0
        self.blink_count = 0
        self.eye_closed_start_time = 0  # Track when eyes started closing
        self.max_eye_closed_duration = settings.BLINK_MAX_CLOSED_SEC  # Configurable max closure time
        
        # Clear any previous EAR smoothing state
        if hasattr(self, '_prev_ear'):
            delattr(self, '_prev_ear')

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
                # Reset blink state if no face detected
                self.blink_state = 0
                self.eye_closed_start_time = 0
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
                
        return results

    def calculate_ear(self, l2d: np.ndarray) -> float:
        """Improved EAR calculation with noise reduction"""
        try:
            # Right eye landmarks (simplified but effective)
            right_eye_top = l2d[386]    # Upper eyelid
            right_eye_bottom = l2d[374]  # Lower eyelid  
            right_eye_left = l2d[362]    # Left corner
            right_eye_right = l2d[263]   # Right corner
            
            # Left eye landmarks
            left_eye_top = l2d[159]     # Upper eyelid
            left_eye_bottom = l2d[145]  # Lower eyelid
            left_eye_left = l2d[33]     # Left corner
            left_eye_right = l2d[133]   # Right corner
            
            # Calculate vertical distances
            right_vertical = np.linalg.norm(right_eye_top - right_eye_bottom)
            left_vertical = np.linalg.norm(left_eye_top - left_eye_bottom)
            
            # Calculate horizontal distances  
            right_horizontal = np.linalg.norm(right_eye_left - right_eye_right)
            left_horizontal = np.linalg.norm(left_eye_left - left_eye_right)
            
            # EAR for each eye (with safety checks)
            if right_horizontal > 0:
                ear_right = right_vertical / right_horizontal
            else:
                ear_right = 0.3  # Default open eye value
                
            if left_horizontal > 0:
                ear_left = left_vertical / left_horizontal
            else:
                ear_left = 0.3  # Default open eye value
            
            # Average both eyes
            ear = (ear_right + ear_left) / 2.0
            
            # Apply simple smoothing to reduce noise (exponential moving average)
            if not hasattr(self, '_prev_ear'):
                self._prev_ear = ear
            else:
                # Smoothing factor - higher value means more smoothing, lower means more responsive.
                # A value of 0.5 provides a balance between noise reduction and responsiveness.
                alpha = 0.5
                ear = alpha * ear + (1 - alpha) * self._prev_ear
                self._prev_ear = ear
            
            return ear
            
        except Exception as e:
            logger.error(f"EAR calculation error: {e}")
            # Return previous EAR if available, otherwise default
            if hasattr(self, '_prev_ear'):
                return self._prev_ear
            return 0.3  # Default open eye

    def detect_blink(self, ear: float) -> bool:
        """
        Detects a blink based on the Eye Aspect Ratio (EAR) using a state machine.

        The process is as follows:
        1.  **State 0 (Eyes Open)**: If the EAR drops below the `ear_threshold`, it transitions
            to State 1, marking the moment the eyes start closing.
        2.  **State 1 (Eyes Closed)**: 
            - If the eyes remain closed for longer than `max_eye_closed_duration`, it's considered
              a false positive (e.g., user is sleeping or looking down), and the state resets.
            - If the EAR rises above the threshold, it signifies the eyes are opening.
            - The duration of the closure is validated to be within a realistic range (150ms - 1s).
            - A debounce check ensures that rapid, successive blinks are not counted.
            - If all checks pass, a blink is registered, and the state resets.

        Args:
            ear (float): The calculated Eye Aspect Ratio.

        Returns:
            bool: True if a valid blink was completed, False otherwise.
        """
        now = time.time()
        blink_completed = False
        
        # Improved state machine with timeout protection
        if self.blink_state == 0:  # Eyes open
            if ear < self.ear_threshold:  # Eyes closing
                self.blink_state = 1
                self.eye_closed_start_time = now  # Track when eyes started closing
                logger.debug(f"Eyes closing - EAR: {ear:.3f}")
                
        elif self.blink_state == 1:  # Eyes closed
            # Check if eyes have been closed too long (possible false detection)
            if now - self.eye_closed_start_time > self.max_eye_closed_duration:
                # Reset state - likely a false detection or user sleeping
                self.blink_state = 0
                self.eye_closed_start_time = 0
                logger.debug(f"Eyes closed too long, resetting - EAR: {ear:.3f}")
            elif ear >= self.ear_threshold:  # Eyes opening
                # Validate blink duration (not too short, not too long)
                blink_duration = now - self.eye_closed_start_time
                if 0.15 <= blink_duration <= 1.0:  # Valid blink duration: 150ms to 1s
                    # Additional debounce check
                    if now - self.last_blink_time > 0.3:  # Minimum 300ms between blinks
                        blink_completed = True
                        self.blink_count += 1
                        self.last_blink_time = now
                        logger.info(f"✅ BLINK DETECTED! EAR: {ear:.3f}, Duration: {blink_duration:.2f}s, Count: {self.blink_count}")
                    else:
                        logger.debug(f"Blink detected but ignored due to debounce - EAR: {ear:.3f}")
                else:
                    logger.debug(f"Blink duration invalid ({blink_duration:.2f}s) - EAR: {ear:.3f}")
                self.blink_state = 0  # Reset to open
                self.eye_closed_start_time = 0
                
        return blink_completed

    def recognize_face(self, face_img):
        """Simple LBPH Recognition"""
        if not self.model_loaded: return -1, 0.0
        
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
        # Basic preprocessing
        face_img = cv2.resize(face_img, (120, 120))
        face_img = cv2.equalizeHist(face_img)
        
        label, confidence = self.recognizer.predict(face_img)
        return label, confidence

    def train_model(self):
        """Simple model training"""
        faces, ids = [], []
        if not os.path.exists(self.faces_dir): os.makedirs(self.faces_dir)
        
        for user_id in os.listdir(self.faces_dir):
            u_path = os.path.join(self.faces_dir, user_id)
            if not os.path.isdir(u_path): continue
            try: uid = int(user_id)
            except: continue
            
            for img_name in os.listdir(u_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        p_img = Image.open(os.path.join(u_path, img_name)).convert('L')
                        img_np = np.array(p_img, 'uint8')
                        img_np = cv2.resize(img_np, (120, 120))
                        img_np = cv2.equalizeHist(img_np)
                        faces.append(img_np)
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
