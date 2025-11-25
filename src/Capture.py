# capture.py
import cv2 as cv

class Capture:
    """
    Class used to capture image frames using the webcam.
    Arguments:
        cam_index: Camera index (0,1,2...) depending on the connected cameras
        mirror: Boolean to mirror the frame horizontally
        rgb: Boolean to convert BGR to RGB (needed for Mediapipe)
    """

    def __init__(self, cam_index: int = 0, mirror: bool = False, rgb: bool = True) -> None:
        self.cam_index = cam_index
        self.mirror = mirror
        self.rgb = rgb

        self.cam = cv.VideoCapture(self.cam_index)
        if not self.cam.isOpened():
            raise Exception(f"Camera with index {self.cam_index} could not be opened.")

    def get_frame(self):
        """
        Returns a single frame.
        Output → (frame_bgr, frame_rgb(optional), success)
        """
        success, frame = self.cam.read()
        if not success:
            return None, None, False

        # Mirror if enabled
        if self.mirror:
            frame = cv.flip(frame, 1)

        # Convert to RGB if enabled
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) if self.rgb else None

        return frame, frame_rgb, True

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Return current frame width and height in pixels as (width, height).
        This uses the camera properties if available, otherwise falls back to reading a frame.
        """
        # Try reading properties first
        fw = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        fh = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        if fw > 0 and fh > 0:
            return fw, fh

        # Fallback: capture one frame and check shape
        ret, frame = self.cam.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return w, h

        # Last resort
        return 640, 480

    def norm_to_pixel(self, nx: float, ny: float) -> Tuple[int, int]:
        """
        Convert normalized coords (0..1) to pixels using current frame size.
        """
        w, h = self.get_frame_size()
        x = int(nx * w)
        y = int(ny * h)
        return x, y

    def pixel_to_norm(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coords to normalized coords (0..1).
        """
        w, h = self.get_frame_size()
        return x / w, y / h


    def release(self):
        """
        Releases the camera.
        """
        self.cam.release()
        cv.destroyAllWindows()
