import cv2
import time
import threading

class Camera:
    def __init__(self, source=0):
        try:
            self.video = cv2.VideoCapture(source)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.video.set(cv2.CAP_PROP_FPS, 15)
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
