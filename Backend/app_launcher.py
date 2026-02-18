import subprocess

def launch_app(command: str):
    try:
        subprocess.Popen(command)
        return True
    except Exception as e:
        print("Error:", e)
        return False
