# milestone1_hand_detection.py
import cv2
import mediapipe as mp

# Initialize Mediapipe modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Create webcam object
cap = cv2.VideoCapture(0)

# Mediapipe Hands object
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Flip frame (mirror effect)
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = hands.process(img_rgb)

        # Draw landmarks if hands detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Display window
        cv2.imshow("Milestone 1 - Hand Detection", frame)

        # Exit when pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
