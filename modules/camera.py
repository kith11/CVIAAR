import cv2
import time
import threading

class Camera:
    def __init__(self, source=0):
        self.video = cv2.VideoCapture(source)
        # Set higher resolution (HD) for better recognition
        # Try 1280x720. If slow, go back to 640x480.
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.lock = threading.Lock()

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        with self.lock:
            if self.video.isOpened():
                success, frame = self.video.read()
                if success:
                    # Resize for performance if needed, but 640x480 is usually okay
                    # frame = cv2.resize(frame, (320, 240))
                    return frame
        return None

    def get_jpeg_frame(self):
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None
