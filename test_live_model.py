import cv2
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 🔹 Load trained components
model = tf.keras.models.load_model("model/gesture_model.h5")
scaler = joblib.load("model/scaler.pkl")
labels = joblib.load("model/labels.pkl")

MODEL_PATH = "vision/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)


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
            gesture_name = labels.inverse_transform([class_id])[0]
            confidence = np.max(prediction)

            cv2.putText(frame,
                        f"{gesture_name} ({confidence:.2f})",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # draw landmarks
            for lm in hand:
                px = int(lm.x * frame.shape[1])
                py = int(lm.y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    cv2.imshow("Live Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
