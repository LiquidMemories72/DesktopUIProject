import ctypes
import os
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui
import requests
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

user32 = ctypes.windll.user32

WINDOW_NAME = "Gesture Controller"

PINCH_CLOSE_THRESHOLD = 0.035
PINCH_OPEN_THRESHOLD = 0.055
CLICK_COOLDOWN = 0.20

SCROLL_PINCH_CLOSE_THRESHOLD = 0.040
SCROLL_PINCH_OPEN_THRESHOLD = 0.060
SCROLL_DEADZONE = 0.004
SCROLL_GAIN = 2200

CONFIDENCE_THRESHOLD = 0.9
HOLD_TIME = 1.5
MOVE_THRESHOLD = 5
FRAME_MARGIN = 120

API_URL = "http://127.0.0.1:8000/trigger/"

POINTER_MODE = False
LAST_MODE_CHECK = 0

pinch_active = False
scroll_pinch_active = False
last_click_time = 0
prev_scroll_y = None

smooth_x, smooth_y = 0, 0
prev_x, prev_y = 0, 0


def set_window_on_top(window_name):
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0013)


def move_mouse(x, y):
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    abs_x = int(x * 65535 / screen_w)
    abs_y = int(y * 65535 / screen_h)

    ctypes.windll.user32.mouse_event(
        0x0001 | 0x8000,
        abs_x,
        abs_y,
        0,
        0,
    )


def pinch_distance(hand, idx_a, idx_b):
    a = hand[idx_a]
    b = hand[idx_b]
    return np.hypot(a.x - b.x, a.y - b.y)


def pick_pointer_and_control_hands(result):
    hands = []
    for i, hand in enumerate(result.hand_landmarks):
        label = "Unknown"
        if result.handedness and i < len(result.handedness) and result.handedness[i]:
            label = result.handedness[i][0].category_name
        hands.append({"label": label, "landmarks": hand})

    pointer_hand = None
    control_hand = None

    for item in hands:
        if item["label"] == "Right":
            pointer_hand = item["landmarks"]
        elif item["label"] == "Left":
            control_hand = item["landmarks"]

    return pointer_hand, control_hand, hands


screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

BASE_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.abspath(os.path.join(BASE_DIR, "model", "gesture_model.h5"))
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "vision", "hand_landmarker.task")

model = tf.keras.models.load_model(MODEL_FILE)
scaler = joblib.load(SCALER_PATH)
labels = joblib.load(LABELS_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)

prediction_buffer = deque(maxlen=10)
candidate_gesture = None
gesture_start_time = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while True:
    now = time.time()
    if now - LAST_MODE_CHECK > 0.5:
        try:
            res = requests.get("http://127.0.0.1:8000/status").json()
            POINTER_MODE = res.get("pointer_mode", False)
        except Exception:
            pass
        LAST_MODE_CHECK = now

    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        pointer_hand, control_hand, hands = pick_pointer_and_control_hands(result)

        if POINTER_MODE:
            # Right Hand for movement
            if pointer_hand is not None:
                index_tip = pointer_hand[8]
                h, w, _ = frame.shape

                raw_x = np.interp(index_tip.x * w, (FRAME_MARGIN, w - FRAME_MARGIN), (0, screen_w))
                raw_y = np.interp(index_tip.y * h, (FRAME_MARGIN, h - FRAME_MARGIN), (0, screen_h))

                distance = np.hypot(raw_x - smooth_x, raw_y - smooth_y)
                if distance < 40:
                    smoothing = 7
                elif distance < 100:
                    smoothing = 5
                else:
                    smoothing = 3

                smooth_x += (raw_x - smooth_x) / smoothing
                smooth_y += (raw_y - smooth_y) / smoothing

                dx = abs(smooth_x - prev_x)
                dy = abs(smooth_y - prev_y)
                if dx > MOVE_THRESHOLD or dy > MOVE_THRESHOLD:
                    move_mouse(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y

            # Left Hand for control (click/scroll)
            if control_hand is not None:
                control_click_pinch = pinch_distance(control_hand, 4, 8)
                control_scroll_pinch = pinch_distance(control_hand, 4, 12)

                if (
                    control_click_pinch < PINCH_CLOSE_THRESHOLD
                    and not pinch_active
                    and (now - last_click_time) > CLICK_COOLDOWN
                ):
                    pyautogui.click()
                    pinch_active = True
                    last_click_time = now
                elif control_click_pinch > PINCH_OPEN_THRESHOLD:
                    pinch_active = False

                if control_scroll_pinch < SCROLL_PINCH_CLOSE_THRESHOLD and not pinch_active:
                    if not scroll_pinch_active:
                        scroll_pinch_active = True
                        prev_scroll_y = control_hand[12].y
                    else:
                        current_scroll_y = control_hand[12].y
                        dy_scroll = current_scroll_y - prev_scroll_y
                        prev_scroll_y = current_scroll_y

                        if abs(dy_scroll) > SCROLL_DEADZONE:
                            scroll_steps = int(-dy_scroll * SCROLL_GAIN)
                            if scroll_steps != 0:
                                pyautogui.scroll(scroll_steps)
                elif control_scroll_pinch > SCROLL_PINCH_OPEN_THRESHOLD:
                    scroll_pinch_active = False
                    prev_scroll_y = None
            else:
                pinch_active = False
                scroll_pinch_active = False
                prev_scroll_y = None

        else:
            pinch_active = False
            scroll_pinch_active = False
            prev_scroll_y = None

            active_hand = result.hand_landmarks[0]
            landmarks = []
            for lm in active_hand:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)
            X = scaler.transform(X)
            prediction = model.predict(X, verbose=0)

            class_id = np.argmax(prediction)
            confidence = float(np.max(prediction))

            prediction_buffer.append(class_id)
            stable_id = max(set(prediction_buffer), key=prediction_buffer.count)
            gesture_name = labels.inverse_transform([stable_id])[0]

            cv2.putText(
                frame,
                f"{gesture_name} ({confidence:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            current_time = time.time()
            if confidence > CONFIDENCE_THRESHOLD:
                if candidate_gesture != gesture_name:
                    candidate_gesture = gesture_name
                    gesture_start_time = current_time

                hold_duration = current_time - gesture_start_time
                progress = min(hold_duration / HOLD_TIME, 1.0)
                cv2.rectangle(frame, (10, 80), (210, 100), (255, 255, 255), 2)
                bar_width = int(progress * 200)
                cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    "Hold to confirm",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if hold_duration >= HOLD_TIME:
                    try:
                        requests.post(API_URL + gesture_name)
                        print("Triggered:", gesture_name)
                    except Exception:
                        print("API failed")
                    candidate_gesture = None
                    gesture_start_time = None
            else:
                candidate_gesture = None
                gesture_start_time = None

        for item in hands:
            hand = item["landmarks"]
            for lm in hand:
                px = int(lm.x * frame.shape[1])
                py = int(lm.y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    mode_text = "POINTER MODE" if POINTER_MODE else "GESTURE MODE"
    mode_color = (255, 255, 0) if POINTER_MODE else (0, 255, 255)
    cv2.putText(frame, mode_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    cv2.imshow(WINDOW_NAME, frame)
    set_window_on_top(WINDOW_NAME)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
