from fastapi import FastAPI
import json
import os
from app_launcher import launch_app
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import glob
import pyautogui
import sys
import subprocess
pointer_mode = False
# helper fns
def get_start_menu_apps():

    paths = [
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
        os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs")
    ]

    apps = {}

    for path in paths:
        for file in glob.glob(path + r"\**\*.lnk", recursive=True):

            name = os.path.splitext(os.path.basename(file))[0]

            key = f"app_{name.lower().replace(' ', '_')}"

            apps[key] = {
                "type": "app",
                "label": name,
                "path": file
            }


    return apps

system_state = {
    "status": "ready",
    "last_action": None
}
controller_process = None

class GestureCreate(BaseModel):
    gesture: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




BASE_DIR = os.path.dirname(__file__)


def load_json(filename):
    with open(os.path.join(BASE_DIR, filename)) as f:
        return json.load(f)

actions = load_json("actions.json")
gesture_map = load_json("gesture_map.json")
#get fns

@app.get("/")
def root():
    return {"status": "Backend running"}
@app.get("/available-apps")
def available_apps():
    return get_start_menu_apps()



@app.get("/actions")
def get_actions():
    apps = get_start_menu_apps()
    combined = {**actions, **apps}
    return combined


@app.get("/status")
def get_status():
    return {**system_state, "pointer_mode": pointer_mode}


@app.get("/gestures")
def get_gestures():
    return gesture_map


#post fns
class GestureMapRequest(BaseModel):
    gesture: str
    action: str
@app.post("/start-detection")
def start_detection():

    global controller_process

    if controller_process is not None:
        return {"status": "already running"}

    script_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "gesture_controller.py")
    )

    controller_process = subprocess.Popen(
        [sys.executable, script_path]
    )

    system_state["status"] = "AI running"

    return {"status": "started"}
@app.post("/stop-detection")
def stop_detection():

    global controller_process

    if controller_process:
        controller_process.terminate()
        controller_process = None

    system_state["status"] = "ready"

    return {"status": "stopped"}


@app.post("/add-gesture")
def add_gesture(data: GestureCreate):

    gesture = data.gesture

    if gesture in gesture_map:
        return {"error": "Gesture already exists"}


    all_actions = {**actions, **get_start_menu_apps()}
    first_action = list(all_actions.keys())[0]


    gesture_map[gesture] = first_action

    with open(os.path.join(BASE_DIR, "gesture_map.json"), "w") as f:
        json.dump(gesture_map, f, indent=4)

    return {"message": f"{gesture} added"}
import subprocess
@app.post("/pointer-mode/{state}")
def set_pointer_mode(state: str):
    global pointer_mode

    pointer_mode = (state == "on")
    return {"pointer_mode": pointer_mode}


@app.post("/capture/{gesture_name}")
def capture_gesture(gesture_name: str):

    system_state["status"] = f"Capturing {gesture_name}"

    script_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "vision", "capture_landmarks.py")
    )

    subprocess.run([
        sys.executable,
        script_path,
        gesture_name
    ])

    system_state["status"] = "ready"

    return {"message": f"{gesture_name} captured"}


@app.post("/train")
def train_model():

    system_state["status"] = "training"

    script_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "train_model.py")
    )

    subprocess.run([sys.executable, script_path])

    system_state["status"] = "ready"

    return {"message": "Model trained"}


@app.post("/delete-gesture/{gesture_name}")
def delete_gesture(gesture_name: str):

    if gesture_name not in gesture_map:
        return {"error": "Gesture not found"}

    del gesture_map[gesture_name]

    with open(os.path.join(BASE_DIR, "gesture_map.json"), "w") as f:
        json.dump(gesture_map, f, indent=4)

    return {"message": f"{gesture_name} deleted"}


@app.post("/map-gesture")
def map_gesture(data: GestureMapRequest):

    gesture = data.gesture
    action = data.action


    all_actions = {**actions, **get_start_menu_apps()}

    if action not in all_actions:
        return {"error": "Invalid action"}



    gesture_map[gesture] = action


    with open(os.path.join(BASE_DIR, "gesture_map.json"), "w") as f:
        json.dump(gesture_map, f, indent=4)

    return {
        "message": f"{gesture} mapped to {action}"
    }



@app.post("/trigger/{gesture_name}")


@app.post("/trigger/{gesture_name}")



def trigger_gesture(gesture_name: str):


    action_id = gesture_map.get(gesture_name)

    if not action_id:
        return {"error": "No action mapped"}


    all_actions = {**actions, **get_start_menu_apps()}

    action = all_actions.get(action_id)

    if not action:
        return {"error": f"Action '{action_id}' not found"}

    try:


        if action["type"] == "app":
            os.startfile(action["path"])

        elif action["type"] == "system":
            pyautogui.hotkey(*action["keys"])


        system_state["last_action"] = f"{gesture_name} → {action['label']}"

        return {"message": f"{action['label']} executed"}

    except Exception as e:
        return {"error": str(e)}
