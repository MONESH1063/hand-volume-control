# milestone3_volume_control.py
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# MediaPipe hands init
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# PyCaw volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

def classify_gesture(dist):
    if dist < 30:
        return "Closed ✊"
    elif dist < 80:
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
            print("Failed to open camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture = "No Hand"
        vol_percent = 0
        dist = 0

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            x1, y1 = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
            x2, y2 = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)

            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            dist = math.hypot(x2 - x1, y2 - y1)
            gesture = classify_gesture(dist)

            # Map distance to volume
            vol_db = np.interp(dist, [30, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol_db, None)
            vol_percent = np.interp(dist, [30, 200], [0, 100])

        # Overlay texts
        cv2.putText(frame, f"Gesture: {gesture}", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Vol: {int(vol_percent)} %", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Dist: {int(dist)} px", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Volume bar
        cv2.rectangle(frame, (50, 180), (85, 430), (255, 255, 255), 2)
        bar = np.interp(vol_percent, [0, 100], [430, 180])
        cv2.rectangle(frame, (50, int(bar)), (85, 430), (0, 255, 0), cv2.FILLED)

        cv2.imshow("Milestone 3 - Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
