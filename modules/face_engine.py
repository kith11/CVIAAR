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
    """Enhanced result structure for Face Landmarker"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks_3d: np.ndarray  # 478 3D landmarks
    landmarks_2d: np.ndarray  # 478 2D landmarks
    face_blendshapes: Optional[List] = None
    facial_transformation_matrix: Optional[np.ndarray] = None
    confidence: float = 0.0
    is_real_face: bool = True  # Liveness detection result

@dataclass
class ThermalMetrics:
    """Thermal management metrics"""
    cpu_temperature: float = 0.0
    processing_time_ms: float = 0.0
    frame_skip_count: int = 0
    thermal_throttle_detected: bool = False
    efficiency_score: float = 1.0

class FaceEngine:
    """
    Consolidated Face Engine providing:
    1. 478 3D Facial Landmarking with temporal filtering.
    2. LBPH Face Recognition.
    3. Advanced 3D Liveness Detection (Anti-spoof).
    4. Thermal Management and Power Optimization.
    """
    
    def __init__(self, 
                 model_path: str = "data/lbph_model.yml", 
                 faces_dir: str = "data/faces",
                 process_interval_ms: int = 200,
                 thermal_threshold: float = 70.0,
                 power_save_mode: bool = True):
        
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

        # MediaPipe Landmarker setup
        self.process_interval_ms = process_interval_ms
        self.last_process_time = 0
        self.thermal_threshold = thermal_threshold
        self.power_save_mode = power_save_mode
        self.current_interval = process_interval_ms
        
        self._initialize_landmarker()
        
        # Thermal & Performance tracking
        self.thermal_metrics = ThermalMetrics()
        self.performance_history = []
        self.max_history_size = 100
        self.quality_level = 1.0
        self.resolution_scale = 1.0
        
        # 3D facial measurements for liveness
        self.facial_measurements = {
            'eye_distance_threshold': (0.3, 0.5),
            'nose_chin_ratio_threshold': (0.4, 0.8),
            'depth_variance_threshold': 0.05,
        }
        
        # Temporal filtering state
        self.temporal_state = {
            'last_landmarks': None,
            'last_bbox': None,
            'temporal_consistency_score': 1.0
        }

    def _initialize_landmarker(self):
        """Initialize MediaPipe Face Landmarker with RunningMode.VIDEO for temporal filtering"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            model_task_path = os.path.join("data", "face_landmarker.task")
            if not os.path.exists(model_task_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, model_task_path)
            
            base_options = python.BaseOptions(model_asset_path=model_task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.use_landmarker = True
            logger.info("Face Landmarker initialized with temporal filtering.")
        except Exception as e:
            logger.warning(f"Face Landmarker init failed: {e}. Falling back to Face Mesh.")
            self.use_landmarker = False
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )

    def get_cpu_temperature(self) -> float:
        """Get CPU temperature for thermal management"""
        try:
            import subprocess
            result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                return float(result.stdout.strip().split('=')[1].split("'")[0])
        except:
            pass
        return 45.0 # Default if check fails

    def should_process_frame(self) -> bool:
        """Check if we should process frame based on interval and thermal status"""
        current_time_ms = time.time() * 1000
        if current_time_ms - self.last_process_time < self.current_interval:
            return False
            
        temp = self.get_cpu_temperature()
        self.thermal_metrics.cpu_temperature = temp
        
        if temp > self.thermal_threshold:
            self.current_interval = min(1000, self.current_interval * 1.2)
            self.quality_level = max(0.5, self.quality_level - 0.1)
            return False # Skip frame to cool down
        else:
            self.current_interval = max(self.process_interval_ms, self.current_interval * 0.9)
            self.quality_level = min(1.0, self.quality_level + 0.05)
            
        self.last_process_time = current_time_ms
        return True

    def process_frame(self, frame: np.ndarray) -> List[FaceLandmarkerResult]:
        """Process frame for landmarks and liveness"""
        if not self.should_process_frame():
            return []
            
        if self.power_save_mode and self.quality_level < 0.8:
            frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

        h, w = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)
        
        results = []
        if self.use_landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.face_landmarks:
                for i, landmarks in enumerate(detection_result.face_landmarks):
                    l3d = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    l2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
                    
                    x_min, x_max = int(np.min(l2d[:, 0])), int(np.max(l2d[:, 0]))
                    y_min, y_max = int(np.min(l2d[:, 1])), int(np.max(l2d[:, 1]))
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    is_real = self._check_liveness_3d(l3d, bbox, w, h)
                    
                    results.append(FaceLandmarkerResult(
                        bbox=bbox, landmarks_3d=l3d, landmarks_2d=l2d,
                        confidence=0.9, is_real_face=is_real
                    ))
        return results

    def _check_liveness_3d(self, l3d, bbox, w, h) -> bool:
        """3D liveness detection logic"""
        # 1. Eye distance
        l_eye = np.mean(l3d[[33, 133, 160, 158, 144, 153]], axis=0)
        r_eye = np.mean(l3d[[362, 263, 385, 387, 373, 380]], axis=0)
        eye_dist = np.linalg.norm(r_eye - l_eye) / max(w, h)
        
        # 2. Nose-chin ratio
        nose = l3d[1]
        chin = l3d[152]
        nose_chin_dist = np.linalg.norm(chin - nose) / bbox[3] if bbox[3] > 0 else 0
        
        # 3. Depth variance
        z_var = np.var(l3d[:, 2])
        
        eye_ok = self.facial_measurements['eye_distance_threshold'][0] <= eye_dist <= self.facial_measurements['eye_distance_threshold'][1]
        ratio_ok = self.facial_measurements['nose_chin_ratio_threshold'][0] <= nose_chin_dist <= self.facial_measurements['nose_chin_ratio_threshold'][1]
        depth_ok = z_var >= self.facial_measurements['depth_variance_threshold']
        
        return eye_ok and ratio_ok and depth_ok

    def recognize_face(self, face_img):
        """LBPH Recognition"""
        if not self.model_loaded: return -1, 0.0
        face_img = cv2.resize(face_img, (100, 100))
        face_img = cv2.equalizeHist(face_img)
        label, confidence = self.recognizer.predict(face_img)
        return label, confidence

    def train_model(self):
        """Train LBPH model from data/faces"""
        faces, ids = [], []
        if not os.path.exists(self.faces_dir): os.makedirs(self.faces_dir)
        for user_id in os.listdir(self.faces_dir):
            u_path = os.path.join(self.faces_dir, user_id)
            if not os.path.isdir(u_path): continue
            try: uid = int(user_id)
            except: continue
            for img_name in os.listdir(u_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    p_img = Image.open(os.path.join(u_path, img_name)).convert('L')
                    img_np = cv2.resize(np.array(p_img, 'uint8'), (100, 100))
                    img_np = cv2.equalizeHist(img_np)
                    faces.append(img_np)
                    ids.append(uid)
        if faces:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write(self.model_path)
            self.model_loaded = True
            return True
        return False

    def reload_model(self):
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            self.model_loaded = True
    
    def get_thermal_status(self) -> Dict[str, Any]:
        return {
            'temperature': self.thermal_metrics.cpu_temperature,
            'interval': self.current_interval,
            'quality': self.quality_level
        }
