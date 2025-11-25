# Detector.py
import cv2 as cv
import mediapipe as mp

class Detector:
    """
    Detector class using MediaPipe Hands.
    Arguments:
        max_hands: Maximum number of hands to detect.
        detection_conf: Minimum confidence value from [0.0, 1.0] the hand detection.
        tracking_conf: Minimum confidence value from [0.0, 1.0] the landmark-tracking.
    """

    def __init__(self, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, rgb_frame):
        """
        Takes RGB frame, returns results + landmarks drawn on the frame
        """
        results = self.hands.process(rgb_frame)
        return results

    def get_landmarks_dict(self, hand_landmarks, frame_wh: Optional[Tuple[int,int]] = None, as_pixel: bool = True) -> Dict[int, Tuple[float, float]]:
        """
        Convert a MediaPipe hand_landmarks to a dict:
            {id: (x, y), ...}
        If as_pixel is True and frame_wh provided, returns pixel coords (int).
        If as_pixel is False or frame_wh is None, returns normalized floats (lm.x, lm.y).
        frame_wh: (width, height)
        """
        lm_dict = {}
        for idx, lm in enumerate(hand_landmarks.landmark):
            if as_pixel and frame_wh is not None:
                fw, fh = frame_wh
                x_px = int(lm.x * fw)
                y_px = int(lm.y * fh)
                lm_dict[idx] = (x_px, y_px)
            else:
                lm_dict[idx] = (lm.x, lm.y)
        return lm_dict

    def get_landmark(self, hand_landmarks, lm_id: int,
                     frame_wh: Optional[Tuple[int,int]] = None,
                     as_pixel: bool = True) -> Optional[Tuple[float, float]]:
        """
        Return a single landmark by id.
        - if as_pixel True and frame_wh provided -> returns (x_px, y_px)
        - else returns normalized (x, y)
        Returns None if lm_id out of range.
        """
        if lm_id < 0 or lm_id >= len(hand_landmarks.landmark):
            return None
        lm = hand_landmarks.landmark[lm_id]
        if as_pixel and frame_wh is not None:
            fw, fh = frame_wh
            return int(lm.x * fw), int(lm.y * fh)
        return lm.x, lm.y

    def close(self):
        self.hands.close()
