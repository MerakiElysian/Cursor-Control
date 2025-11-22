# main.py
from Capture import Capture
from Detector import Detector
import cv2

def main():
    cam = Capture(cam_index=0, mirror=True, rgb=True)
    detect = Detector()

    while True:
        frame_bgr, frame_rgb, success = cam.get_frame()

        if not success:
            print("Failed to get frame")
            break

        # Send RGB frame to Mediapipe
        results = detect.detect(frame_rgb)

        # Draw landmarks (if detected)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                detect.draw(frame_bgr, hand_landmarks)

        # Show output
        cv2.imshow("Hand Detection", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    detect.close()


if __name__ == "__main__":
    main()
