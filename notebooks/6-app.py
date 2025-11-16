import cv2
import joblib
import numpy as np
import mediapipe as mp


# TODO !! CHANGE THIS WITH YOUR OWN PATHS (it will have to look similiar to mine, relative paths don't work) !!
SCALER_PATH = "C:\\Users\\kyana\\Documents\\GitHub\\sign-language-alphabet-recognition\\model\\scaler.pkl"
BEST_MODEL_PATH = "C:\\Users\\kyana\\Documents\\GitHub\\sign-language-alphabet-recognition\\model\\neural_network_model.pkl"

scaler = joblib.load(SCALER_PATH)
best_model = joblib.load(BEST_MODEL_PATH)


def predict_sign_language(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    landmarks = np.array(landmarks)
    
    landmarks_reshaped = landmarks.reshape(1, 21, 3)
    

    def center_hand(landmarks_batch):
        wrist = landmarks_batch[0]  # First landmark is wrist
        centered = landmarks_batch - wrist
        return centered
    
    landmarks_centered = center_hand(landmarks_reshaped[0])
    

    def scale_hand(landmarks_batch):
        wrist = landmarks_batch[0]
        middle_finger_tip = landmarks_batch[12]
        scale = np.linalg.norm(middle_finger_tip - wrist)
        
        if scale > 0:
            scaled = landmarks_batch / scale
        else:
            scaled = landmarks_batch
            
        return scaled
    
    landmarks_scaled = scale_hand(landmarks_centered)
    landmarks_flat = landmarks_scaled.flatten()
    landmarks_normalized = scaler.transform([landmarks_flat])[0]
    
    prediction = best_model.predict([landmarks_normalized])[0]
    
    try:
        confidence = np.max(best_model.predict_proba([landmarks_normalized])[0])
        return prediction, confidence
    except:
        return prediction, 1.0
    

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera detected and ready!")
    cap.release()
else:
    print("Camera not found! Please check your webcam connection.")


cap = cv2.VideoCapture(0)
detected_letters = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    frame_count = 0
    detection_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
            
        frame_count += 1
        
        # Mirror the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = hands.process(rgb_frame)
        
        # Default values
        prediction = ""
        confidence = 0.0
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                try:
                    # Make prediction using our function
                    prediction, confidence = predict_sign_language(hand_landmarks)
                    detection_count += 1
                    
                except Exception as e:
                    prediction = "ERROR"
                    confidence = 0.0
        
        # Display prediction on frame
        if prediction and prediction != "ERROR":
            # Main prediction text
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255) if confidence > 0.6 else (0, 0, 255)
            cv2.putText(frame, f"Letter: {prediction}", (10, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

            if prediction not in detected_letters:
                detected_letters.append(prediction)

            # Confidence text
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        else:
            # No detection message
            cv2.putText(frame, "Show your hand clearly", (10, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

        # Statistics
        cv2.putText(frame, f"Frames: {frame_count} | Detections: {detection_count}", 
                   (10, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Instructions
        cv2.putText(frame, "Press ESC to quit", (10, h - 10),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

        # Show the frame
        cv2.imshow("Sign Language Alphabet Detection", frame)
        
        # Break on ESC key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"Detection session ended :)")
print(f"Total frames: {frame_count}")
print(f"Successful detections: {detection_count}")
print(f"Detection rate: {detection_count/max(frame_count,1)*100:.1f}%")
detected_letters = sorted(set(detected_letters))
print("Detected letters in alphabetical order: " + ", ".join(detected_letters))