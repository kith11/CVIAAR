import cv2
import time
import threading
from config import settings

class Camera:
    """
    A thread-safe camera class for capturing video frames from a hardware device.

    This class handles the initialization of the camera, setting its properties (width, height, FPS),
    and provides a thread-safe method to retrieve the latest frame.
    """
    def __init__(self, source=0):
        """
        Initializes the camera.

        Args:
            source (int, optional): The camera source index. Defaults to 0.
        """
        try:
            self.video = cv2.VideoCapture(source)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
            self.video.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
        except Exception as e:
            print(f"Warning: Could not initialize camera: {e}")
            self.video = None
        self.lock = threading.Lock() # Lock to ensure thread-safe frame access

    def __del__(self):
        """Releases the camera resource when the object is deleted."""
        if self.video and self.video.isOpened():
            self.video.release()

    def get_frame(self):
        """
        Retrieves the latest frame from the camera in a thread-safe manner.

        Returns:
            np.ndarray: The captured frame, or None if an error occurs.
        """
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
        """
        Retrieves the latest frame from the camera and encodes it as a JPEG.

        Returns:
            bytes: The JPEG-encoded frame, or None if an error occurs.
        """
        frame = self.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        return None
