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

    def draw(self, frame_bgr, hand_landmarks):
        """
        Draws landmarks on BGR frame
        """
        self.mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def close(self):
        self.hands.close()
