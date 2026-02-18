import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf
import requests
import time
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 🔹 CONFIG
API_URL = "http://127.0.0.1:8000/trigger/"
CONFIDENCE_THRESHOLD = 0.9
COOLDOWN = 2  # seconds


# 🔹 Load model assets
BASE_DIR = os.path.dirname(__file__)

model= os.path.abspath(
    os.path.join(BASE_DIR, "model", "gesture_model.h5")
)

SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.pkl")


MODEL_PATH = os.path.join(BASE_DIR, "vision", "hand_landmarker.task")
model = tf.keras.models.load_model(model)
scaler = joblib.load(SCALER_PATH)
labels = joblib.load(LABELS_PATH)


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)


# 🔹 State control
prediction_buffer = deque(maxlen=10)
last_triggered = None
last_trigger_time = 0


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            landmarks = []

            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)
            X = scaler.transform(X)

            prediction = model.predict(X, verbose=0)

            class_id = np.argmax(prediction)
            confidence = np.max(prediction)

            prediction_buffer.append(class_id)

            stable_id = max(set(prediction_buffer), key=prediction_buffer.count)
            gesture_name = labels.inverse_transform([stable_id])[0]

            # UI
            cv2.putText(frame,
                        f"{gesture_name} ({confidence:.2f})",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # 🔥 TRIGGER LOGIC
            current_time = time.time()

            if (confidence > CONFIDENCE_THRESHOLD and
                gesture_name != last_triggered and
                current_time - last_trigger_time > COOLDOWN):

                try:
                    requests.post(API_URL + gesture_name)
                    print(f"Triggered: {gesture_name}")

                    last_triggered = gesture_name
                    last_trigger_time = current_time

                except:
                    print("API call failed")

            # draw landmarks
            for lm in hand:
                px = int(lm.x * frame.shape[1])
                py = int(lm.y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
