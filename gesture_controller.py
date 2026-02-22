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
import pyautogui
import ctypes

user32 = ctypes.windll.user32


def set_window_on_top(window_name):
    """Set OpenCV window to always stay on top"""
    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:

        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0003)
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
        0
    )
def mouse_down():
    ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)

def mouse_up():
    ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
pinch_active = False
PINCH_CLOSE_THRESHOLD = 0.035
PINCH_OPEN_THRESHOLD = 0.055
CLICK_COOLDOWN = 0.20
last_click_time = 0
POINTER_MODE = False
LAST_MODE_CHECK = 0





API_URL = "http://127.0.0.1:8000/trigger/"
CONFIDENCE_THRESHOLD = 0.9
HOLD_TIME = 1.5
smooth_x, smooth_y = 0, 0
prev_x, prev_y = 0, 0

SMOOTHING = 8
MOVE_THRESHOLD = 5



screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False
FRAME_MARGIN = 120





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
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)



prediction_buffer = deque(maxlen=10)

candidate_gesture = None
gesture_start_time = None



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)




while True:

    if time.time() - LAST_MODE_CHECK > 0.5:
        try:
            res = requests.get("http://127.0.0.1:8000/status").json()
            POINTER_MODE = res.get("pointer_mode", False)
        except:
            pass
        LAST_MODE_CHECK = time.time()




    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:


        for hand in result.hand_landmarks:
            thumb_tip = hand[4]
            index_tip = hand[8]

            pinch_distance = np.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

















            h, w, _ = frame.shape

            raw_x = np.interp(index_tip.x * w,
                            (FRAME_MARGIN, w - FRAME_MARGIN),
                            (0, screen_w))

            raw_y = np.interp(index_tip.y * h,
                            (FRAME_MARGIN, h - FRAME_MARGIN),
                            (0, screen_h))


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

            if POINTER_MODE and (dx > MOVE_THRESHOLD or dy > MOVE_THRESHOLD):
                move_mouse(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y






            landmarks = []

            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)
            X = scaler.transform(X)
            if not POINTER_MODE:
                prediction = model.predict(X, verbose=0)

                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                prediction_buffer.append(class_id)

                stable_id = max(set(prediction_buffer), key=prediction_buffer.count)
                gesture_name = labels.inverse_transform([stable_id])[0]
                cv2.putText(frame,
                        f"{gesture_name} ({confidence:.2f})",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)






            current_time = time.time()

            if not POINTER_MODE and confidence > CONFIDENCE_THRESHOLD:


                if candidate_gesture != gesture_name:
                    candidate_gesture = gesture_name
                    gesture_start_time = current_time

                hold_duration = current_time - gesture_start_time
                progress = min(hold_duration / HOLD_TIME, 1.0)


                cv2.rectangle(frame, (10, 80), (210, 100), (255, 255, 255), 2)


                bar_width = int(progress * 200)
                cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), (0, 255, 0), -1)

                cv2.putText(frame, "Hold to confirm", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if hold_duration >= HOLD_TIME:
                    try:
                        requests.post(API_URL + gesture_name)
                        print("Triggered:", gesture_name)
                    except:
                        print("API failed")

                    candidate_gesture = None
                    gesture_start_time = None

            else:
                candidate_gesture = None
                gesture_start_time = None



            for lm in hand:
                px = int(lm.x * frame.shape[1])
                py = int(lm.y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    mode_text = "POINTER MODE" if POINTER_MODE else "GESTURE MODE"
    mode_color = (255,255,0) if POINTER_MODE else (0,255,255)

    cv2.putText(frame, mode_text, (10,140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    cv2.imshow("Gesture Controller", frame)
    set_window_on_top("Gesture Controller")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
