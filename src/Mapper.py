# Mapper.py
import time
import cv2 as cv
import mediapipe as mp

class Mapper:
    """
    Mapper draws MediaPipe hand landmarks and an FPS counter on BGR frames.

    Usage:
        mapper = Mapper(show_fps=True)
        mapper.draw(frame_bgr, hand_landmarks)   # call for each detected hand
    """

    def __init__(self, show_fps: bool = True, fps_position: tuple = (10, 30), fps_scale: float = 0.8):
        self.show_fps = show_fps
        self.fps_position = fps_position
        self.fps_scale = fps_scale

        # MediaPipe helpers
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # FPS tracking
        self._prev_time = time.time()
        self._fps = 0.0

        # style for putText
        self._font = cv.FONT_HERSHEY_DUPLEX
        self._thickness = 2
        self._bg_padding = 6

    def _update_fps(self):
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            # simple moving update (smoothed a little)
            instant_fps = 1.0 / dt
            # low-pass smoothing
            self._fps = 0.85 * self._fps + 0.15 * instant_fps if self._fps != 0 else instant_fps
        self._prev_time = now

    def _draw_fps(self, frame):
        text = f"FPS: {int(self._fps)}"
        x, y = self.fps_position

        # get text size for background rectangle
        (w, h), _ = cv.getTextSize(text, self._font, self.fps_scale, self._thickness)
        # background rectangle
        cv.rectangle(frame, (x - 4, y - h - 4), (x + w + 4, y + 6), (30, 30, 30), cv.FILLED)
        # put text (white on top)
        cv.putText(frame, text, (x, y), self._font, self.fps_scale, (0, 0, 200), self._thickness, cv.LINE_AA)

    def draw(self, frame_bgr, hand_landmarks):
        """
        Draws landmarks for one hand (hand_landmarks) on the provided BGR frame.
        Call this once per detected hand. If show_fps=True, it updates and draws FPS once per call.
        """
        # draw landmarks + connections
        self.mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_draw.DrawingSpec(color=(100,100,0), thickness=2, circle_radius=1)
        )

        # update + draw fps (you can call it every frame; calling it multiple times same frame is harmless)
        if self.show_fps:
            self._update_fps()
            self._draw_fps(frame_bgr)
