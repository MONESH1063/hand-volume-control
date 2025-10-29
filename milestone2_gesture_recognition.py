# milestone2_gesture_recognition.py
import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def classify_gesture(distance):
    if distance < 30:
        return "Closed ✊"
    elif distance < 80:
        return "Pinch 🤏"
    else:
        return "Open 🖐"

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture_text = "No Hand"
        distance_text = "-"

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                h, w, _ = frame.shape

                # Landmark points for thumb-tip (4) and index-tip (8)
                x1, y1 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
                x2, y2 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)

                # Draw circles & line
                cv2.circle(frame, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Compute Euclidean distance
                distance = math.hypot(x2 - x1, y2 - y1)
                distance_text = f"{int(distance)} px"
                gesture_text = classify_gesture(distance)

        # Overlay info on frame
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Distance: {distance_text}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Milestone 2 – Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
