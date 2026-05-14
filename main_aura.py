from picamera2 import Picamera2
import cv2
import threading
import time
import os
import json
import subprocess # Replaces pyttsx3 to save 120MB of RAM
from vosk import Model, KaldiRecognizer
import wave

# ── Constants ──────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 800, 480
CASCADE_PATH = "/home/aura/AURA/model/haarcascade_frontalface_default.xml"
MODEL_PATH   = "/home/aura/AURA/model/vosk-model-small-en-in-0.4"

COMMANDS = {
    "forward":  "move_forward",
    "back":     "move_backward",
    "left":     "turn_left",
    "right":    "turn_right",
    "stop":     "stop_motors",
}

# ── Shared state ───────────────────────────────────────
state = {
    "faces":          [],
    "heard_text":     "",
    "response_text":  "",
    "status":         "Initializing...",
    "greeted":        False,   
    "face_lost_time": None,    
}
state_lock = threading.Lock()

# ── Optimized Ultra-Lightweight TTS ─────────────────────
def speak(text):
    """
    Spawns an independent OS process for espeak.
    Uses 0MB of Python script RAM and cannot lock OpenCV frames.
    """
    def _speak_task():
        # -s 150 adjusts the speaking speed rate to matching your original code
        subprocess.run(["espeak", "-s", "150", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=_speak_task, daemon=True).start()

# ── Motor control (stub) ──────────────────────────────
def execute_command(action):
    print(f"Motor: {action}")

# ── Face greeting logic ────────────────────────────────
def handle_greeting(face_count):
    with state_lock:
        greeted    = state["greeted"]
        lost_time  = state["face_lost_time"]

    if face_count > 0:
        with state_lock:
            state["face_lost_time"] = None

        if not greeted:
            with state_lock:
                state["greeted"]       = True
                state["response_text"] = "Hello! I am AURA"
            speak("Hello! I am AURA")
    else:
        now = time.time()
        with state_lock:
            if state["face_lost_time"] is None:
                state["face_lost_time"] = now
            lost_since = now - state["face_lost_time"]

        if lost_since > 3.0:
            with state_lock:
                state["greeted"]       = False
                state["response_text"] = ""

# ── Voice thread (Optimized for 1GB RAM) ───────────────
def voice_thread():
    time.sleep(2)  # Wait for camera to boot up completely

    # Initialize Vosk Model
    model = Model(MODEL_PATH)
    rec   = KaldiRecognizer(model, 16000)

    with state_lock:
        state["status"] = "Listening..."

    wf = wave.open("/home/aura/AURA/test_input.wav", "rb")

    while True:
        # Lower frame buffer read step to prevent locking CPU cycles
        data = wf.readframes(2000) 
        if len(data) == 0:
            wf.rewind()           
            rec = KaldiRecognizer(model, 16000)
            time.sleep(1) # Yield core execution to OpenCV
            continue

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text   = result.get("text", "").lower().strip()

            if text:
                with state_lock:
                    state["heard_text"] = f"Heard: {text}"

                matched = None
                for keyword, action in COMMANDS.items():
                    if keyword in text:
                        matched = action
                        break

                if matched:
                    response = matched.replace("_", " ")
                    with state_lock:
                        state["response_text"] = f"CMD: {response}"
                    speak(response)
                    execute_command(matched)
                else:
                    with state_lock:
                        state["response_text"] = "Unknown command"
                    speak("Command not recognized")

# ── Draw overlay on frame ──────────────────────────────
def draw_overlay(frame, faces, heard, response, status):
    h, w = frame.shape[:2]

    # Face Boxes
    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Semi-transparent bottom status window bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if heard:
        cv2.putText(frame, heard, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    if response:
        cv2.putText(frame, response, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, status, (w - 160, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    face_label = f"Faces: {len(faces)}"
    cv2.putText(frame, face_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

    return frame

# ── Camera + display (main thread) ────────────────────
def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    picam2 = Picamera2()
    # Force 320x240 processing resolution to maximize available memory space
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # Create window and scale cleanly to match display screen size
    cv2.namedWindow("AURA", cv2.WINDOW_NORMAL)
    cv2.moveWindow("AURA", 0, 0)
    cv2.resizeWindow("AURA", DISPLAY_W, DISPLAY_H)
    cv2.setWindowProperty("AURA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start Voice Processing Thread
    t = threading.Thread(target=voice_thread, daemon=True)
    t.start()

    with state_lock:
        state["status"] = "AURA Ready"

    frame_count = 0
    local_faces = [] # Local loop buffer variable to minimize state_lock read operations

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame_count % 5 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Tweaked processing scaleFactor for 32-bit hardware efficiency
            detected_boxes = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3,
                minNeighbors=5, minSize=(30, 30)
            )
            
            # Map detected array tuples efficiently
            local_faces = [tuple(b) for b in detected_boxes] if len(detected_boxes) > 0 else []
            
            with state_lock:
                state["faces"] = local_faces

            handle_greeting(len(local_faces))
            frame_count = 0
        frame_count += 1

        # Atomically snap background changes
        with state_lock:
            heard    = state["heard_text"]
            response = state["response_text"]
            status   = state["status"]

        # Run UI layout overlays
        frame = draw_overlay(frame, local_faces, heard, response, status)

        # Upscale frame smoothly using hardware acceleration to fit the monitor output
        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("AURA", display_frame)

        # 25ms delay ensures CPU cores can breathe on 1GB configurations
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()

