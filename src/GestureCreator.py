import cv2
import mediapipe as mp
import numpy as np
import json
from collections import defaultdict

class CustomGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Storage for gesture templates
        self.gesture_templates = {}
        self.recording_samples = []
        self.current_gesture_name = None
        
    def extract_features(self, landmarks):
        """Extract normalized features from hand landmarks"""
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Normalize relative to wrist (landmark 0)
        wrist = points[0]
        normalized = points - wrist
        
        # Calculate finger angles and distances
        features = []
        
        # Finger tip distances from wrist
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        for tip in fingertips:
            dist = np.linalg.norm(normalized[tip])
            features.append(dist)
        
        # Angles between consecutive fingers
        for i in range(len(fingertips) - 1):
            vec1 = normalized[fingertips[i]]
            vec2 = normalized[fingertips[i+1]]
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            features.append(cos_angle)
        
        # Finger curl (distance between fingertip and base)
        finger_bases = [2, 5, 9, 13, 17]
        for base, tip in zip(finger_bases, fingertips):
            curl = np.linalg.norm(normalized[tip] - normalized[base])
            features.append(curl)
        
        return np.array(features)
    
    def start_recording(self, gesture_name, num_samples=3):
        """Start recording samples for a new gesture"""
        self.current_gesture_name = gesture_name
        self.recording_samples = []
        print(f"Recording gesture '{gesture_name}'. Press SPACE to capture {num_samples} samples.")
        return num_samples
    
    def add_sample(self, landmarks):
        """Add a sample during recording"""
        features = self.extract_features(landmarks)
        self.recording_samples.append(features)
        print(f"Sample {len(self.recording_samples)} captured!")
        return len(self.recording_samples)
    
    def finalize_gesture(self):
        """Calculate the template from recorded samples"""
        if len(self.recording_samples) < 2:
            print("Need at least 2 samples!")
            return False
        
        # Calculate mean and standard deviation (the "essence")
        samples_array = np.array(self.recording_samples)
        mean_features = np.mean(samples_array, axis=0)
        std_features = np.std(samples_array, axis=0)
        
        # Store template with tolerance based on variation
        self.gesture_templates[self.current_gesture_name] = {
            'mean': mean_features.tolist(),
            'std': std_features.tolist(),
            'samples': samples_array.tolist()
        }
        
        print(f"Gesture '{self.current_gesture_name}' saved with {len(self.recording_samples)} samples!")
        self.recording_samples = []
        self.current_gesture_name = None
        return True
    
    def recognize_gesture(self, landmarks, threshold=0.85):
        """Recognize gesture from current hand landmarks"""
        if not self.gesture_templates:
            return None, 0.0
        
        current_features = self.extract_features(landmarks)
        
        best_match = None
        best_similarity = 0.0
        
        for gesture_name, template in self.gesture_templates.items():
            mean = np.array(template['mean'])
            std = np.array(template['std'])
            
            # Cosine similarity
            cos_sim = np.dot(current_features, mean) / (
                np.linalg.norm(current_features) * np.linalg.norm(mean) + 1e-6
            )
            
            # Weighted by inverse of std (more weight to stable features)
            weights = 1.0 / (std + 0.1)
            weighted_diff = np.abs(current_features - mean) * weights
            distance_score = 1.0 / (1.0 + np.mean(weighted_diff))
            
            # Combined similarity
            similarity = (cos_sim + distance_score) / 2.0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = gesture_name
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        return None, best_similarity
    
    def save_gestures(self, filename='gestures.json'):
        """Save gesture templates to file"""
        with open(filename, 'w') as f:
            json.dump(self.gesture_templates, f, indent=2)
        print(f"Gestures saved to {filename}")
    
    def load_gestures(self, filename='gestures.json'):
        """Load gesture templates from file"""
        try:
            with open(filename, 'r') as f:
                self.gesture_templates = json.load(f)
            print(f"Loaded {len(self.gesture_templates)} gestures from {filename}")
            return True
        except FileNotFoundError:
            print(f"No saved gestures found at {filename}")
            return False


def main():
    recognizer = CustomGestureRecognizer()
    recognizer.load_gestures()  # Load existing gestures if available
    
    cap = cv2.VideoCapture(0)
    
    mode = 'recognize'  # 'record' or 'recognize'
    recording_gesture = None
    samples_needed = 0
    samples_captured = 0
    
    print("\n=== Custom Gesture Recognition ===")
    print("Press 'r' to start recording a new gesture")
    print("Press 's' to save gestures")
    print("Press 'q' to quit")
    print("Press SPACE to capture samples during recording\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                recognizer.mp_draw.draw_landmarks(
                    frame, hand_landmarks, recognizer.mp_hands.HAND_CONNECTIONS
                )
                
                if mode == 'recognize':
                    gesture, confidence = recognizer.recognize_gesture(hand_landmarks)
                    if gesture:
                        cv2.putText(frame, f"{gesture}: {confidence:.2f}", 
                                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2)
                        # Here you can trigger your task based on the gesture
                        # perform_task(gesture)
        
        # Display mode and instructions
        if mode == 'record':
            cv2.putText(frame, f"Recording: {recording_gesture} ({samples_captured}/{samples_needed})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Mode: Recognition", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Custom Gesture Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r') and mode == 'recognize':
            gesture_name = input("\nEnter gesture name: ")
            samples_needed = recognizer.start_recording(gesture_name, num_samples=3)
            samples_captured = 0
            mode = 'record'
            recording_gesture = gesture_name
        elif key == ord(' ') and mode == 'record' and results.multi_hand_landmarks:
            samples_captured = recognizer.add_sample(results.multi_hand_landmarks[0])
            if samples_captured >= samples_needed:
                recognizer.finalize_gesture()
                mode = 'recognize'
                recording_gesture = None
        elif key == ord('s'):
            recognizer.save_gestures()
    
    cap.release()
    cv2.destroyAllWindows()
    recognizer.hands.close()


if __name__ == "__main__":
    main()