import cv2
import time
import csv
import os
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
os.makedirs(DATASET_PATH, exist_ok=True)

CAPTURE_TIME = 25
DOT_RADIUS = 12
MOVE_SPEED = 3   # lower = slower smoother


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)


def save_landmarks(landmarks, gesture_name):

    os.makedirs(DATASET_PATH, exist_ok=True)
    file = os.path.join(DATASET_PATH, f"{gesture_name}.csv")

    row = []

    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])

    row.append(gesture_name)

    with open(file, "a", newline="") as f:
        csv.writer(f).writerow(row)


def capture_gesture(gesture_name):

    cap = cv2.VideoCapture(0)

    positions = [
        np.array((100, 100)),
        np.array((540, 100)),
        np.array((100, 380)),
        np.array((540, 380)),
        np.array((320, 240))
    ]

    pos_index = 0
    current_pos = positions[0].astype(float)
    target_pos = positions[1]

    # 🟢 READY SCREEN
    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Gesture: {gesture_name}", (180, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, "Press SPACE when ready", (140, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) == 32:
            break

    start_time = time.time()

    # 🔵 CAPTURE LOOP
    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = int(CAPTURE_TIME - elapsed)

        if remaining <= 0:
            break

        # smooth dot motion
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 5:
            pos_index = (pos_index + 1) % len(positions)
            target_pos = positions[pos_index]
        else:
            current_pos += direction / distance * MOVE_SPEED

        cx, cy = current_pos.astype(int)

        # 🔥 MediaPipe on FULL frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:

                save_landmarks(hand, gesture_name)

                for lm in hand:
                    px = int(lm.x * frame.shape[1])
                    py = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        # 🎯 guiding dot
        cv2.circle(frame, (cx, cy), DOT_RADIUS, (0, 0, 255), -1)

        # UI text
        cv2.putText(frame, f"Time left: {remaining}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, "Follow the dot with your hand",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Capture complete for {gesture_name}")
import sys

if __name__ == "__main__":
    gesture = sys.argv[1]
    capture_gesture(gesture)
