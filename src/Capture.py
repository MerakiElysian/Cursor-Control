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

    def release(self):
        """
        Releases the camera.
        """
        self.cam.release()
        cv.destroyAllWindows()
