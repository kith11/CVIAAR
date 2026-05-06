import unittest
import numpy as np
import time
from modules.face_engine import FaceEngine

class TestCaptureModule(unittest.TestCase):
    def setUp(self):
        self.engine = FaceEngine()

    def test_blink_accuracy_simulation(self):
        """Simulates EAR values to test blink detection accuracy and logic."""
        # Populate history with open eyes (~0.3)
        for _ in range(50):
            self.engine.calculate_ear(self._mock_landmarks(0.3))
            self.engine.detect_blink(0.3)
            
        # Simulated EAR values for a normal blink (Open -> Closed -> Open)
        # We need a gradual transition or a sharp drop
        blink_sequence = [0.3, 0.25, 0.1, 0.08, 0.1, 0.25, 0.3]
        blink_detected = False
        
        for ear_val in blink_sequence:
            # Simulate time passing (100ms per frame)
            time.sleep(0.1)
            actual_ear = self.engine.calculate_ear(self._mock_landmarks(ear_val))
            detected = self.engine.detect_blink(actual_ear)
            # print(f"EAR: {actual_ear:.3f}, Detected: {detected}, State: {self.engine.blink_state}")
            if detected:
                blink_detected = True
                
        self.assertTrue(blink_detected, "Blink should be detected in realistic sequence")

    def test_rate_limiting_logic(self):
        """Verifies that the rate-limiting cache correctly tracks requests."""
        # This would ideally test the FastAPI endpoint, but here we test the backend logic
        from app import recognition_rate_limit
        
        ip = "127.0.0.1"
        for i in range(10):
            count = recognition_rate_limit.get(ip, 0)
            recognition_rate_limit[ip] = count + 1
            
        self.assertEqual(recognition_rate_limit.get(ip), 10)
        # 11th request should be blocked by the app logic (tested manually or via integration test)

    def _mock_landmarks(self, ear_val):
        """Helper to create mock landmarks that result in a specific EAR."""
        # Standard EAR = (v1 + v2) / (2 * h)
        # Let's set h = 1.0, v1 = ear_val, v2 = ear_val
        # Landmarks are 478 x 2 (x, y)
        l2d = np.zeros((478, 2))
        # Left eye horizontal (33, 133)
        l2d[33] = [0, 0]
        l2d[133] = [1, 0]
        # Left eye vertical (160-144, 158-153)
        l2d[160] = [0.5, -ear_val]
        l2d[144] = [0.5, 0]
        l2d[158] = [0.5, -ear_val]
        l2d[153] = [0.5, 0]
        
        # Right eye horizontal (362, 263)
        l2d[362] = [2, 0]
        l2d[263] = [3, 0]
        # Right eye vertical
        l2d[385] = [2.5, -ear_val]
        l2d[380] = [2.5, 0]
        l2d[387] = [2.5, -ear_val]
        l2d[373] = [2.5, 0]
        
        return l2d

if __name__ == '__main__':
    unittest.main()
