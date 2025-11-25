# main.py
from Capture import Capture
from Detector import Detector
from Mapper import Mapper
import cv2 as cv

def main():
    cam = Capture(cam_index=0, mirror=True, rgb=True)
    detect = Detector()
    mapper = Mapper()

    while True:
        frame_bgr, frame_rgb, success = cam.get_frame()
        
        if not success:
            print("Failed to get frame")
            break
        mapper._update_fps()
        # Send RGB frame to Mediapipe
        results = detect.detect(frame_rgb)

        # Draw landmarks (if detected)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # To print the positional co-ordinates for points on the hands.
                for id, lm in enumerate(hand_landmarks.landmark):
                    print(id,lm) # Print the x,y coordinates of landmarks =
                mapper.draw(frame_bgr, hand_landmarks)

        # Show output
        cv.imshow("Hand Detection", frame_bgr)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    detect.close()


if __name__ == "__main__":
    main()
