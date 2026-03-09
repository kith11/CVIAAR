import cv2
import time
import threading
from config import settings

class Camera:
    def __init__(self, source=0):
        try:
            self.video = cv2.VideoCapture(source)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
            self.video.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
        except Exception as e:
            print(f"Warning: Could not initialize camera: {e}")
            self.video = None
        self.lock = threading.Lock()

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()

    def get_frame(self):
        with self.lock:
            if self.video and self.video.isOpened():
                try:
                    success, frame = self.video.read()
                    if success:
                        return frame
                except:
                    pass
        return None

    def get_jpeg_frame(self):
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None
