import pyaudio  

from picamera2 import Picamera2
import cv2
import threading
import time
import os
import json
import subprocess 
from vosk import Model, KaldiRecognizer
import wave

# ── Constants ──────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 800, 480
CASCADE_PATH = "/home/aura/AURA/model/haarcascade_frontalface_default.xml"
MODEL_PATH   = "/home/aura/AURA/model/vosk-model-small-en-in-0.4"
PIPE_PATH    = "/home/aura/AURA/ssh_bridge.pipe"

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
    "speaking":       False,    
    "face_lost_time": None,    
}
state_lock = threading.Lock()

# ── Centralized Execution & Screen Management ─────────
def process_text_command(text_input, source_type="Input"):
    """
    Unified text matching logic. Overwrites old text on the LCD, 
    speaks via espeak, and wipes the display clear when talking finishes.
    """
    if not text_input:
        return

    # 1. Instantly overwrite old text responses and push new input onto LCD
    with state_lock:
        state["heard_text"] = f"{source_type}: {text_input}"

    matched = None
    for keyword, action in COMMANDS.items():
        if keyword in text_input:
            matched = action
            break

    if matched:
        response = matched.replace("_", " ")
        with state_lock:
            state["response_text"] = f"CMD: {response}"
        
        # Lock camera greeting execution during action speech
        with state_lock:
            state["speaking"] = True
        
        # Synchronous audio processing
        subprocess.run(["espeak", "-s", "150", response], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with state_lock:
            state["speaking"] = False
            # Clear text arrays immediately after speaking finishes
            state["heard_text"] = ""
            state["response_text"] = ""
            
        execute_command(matched)
    else:
        with state_lock:
            state["response_text"] = "Unknown command"
        
        with state_lock:
            state["speaking"] = True
            
        subprocess.run(["espeak", "-s", "150", "Command not recognized"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with state_lock:
            state["speaking"] = False
            # Clear text error displays immediately
            state["heard_text"] = ""
            state["response_text"] = ""

# ── Optimized Ultra-Lightweight TTS ─────────────────────
def speak(text):
    def _speak_task():
        with state_lock:
            state["speaking"] = True
        subprocess.run(["espeak", "-s", "150", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with state_lock:
            state["speaking"] = False
    threading.Thread(target=_speak_task, daemon=True).start()

def handle_greeting(face_count):
    with state_lock:
        greeted    = state["greeted"]
        speaking   = state["speaking"]  
        lost_time  = state["face_lost_time"]

    if face_count > 0:
        with state_lock:
            state["face_lost_time"] = None

        if not greeted and not speaking:
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

# ── Voice thread (Active Output Tracker) ───────────────
def voice_thread():
    time.sleep(2)  

    try:
        model = Model(MODEL_PATH)
        rec   = KaldiRecognizer(model, 16000)
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=16000,
            input=True, frames_per_buffer=4000  
        )
    except Exception as e:
        # Graceful microphone detection bypass so your system does not crash without a mic
        with state_lock:
            state["status"] = "SSH Only Mode"
        print(f"⚠️ Microphone offline or missing: {e}. Remote SSH mode active.")
        while True:
            time.sleep(1)

    with state_lock:
        state["status"] = "AURA Listening"

    while True:
        try:
            data = stream.read(4000, exception_on_overflow=False)
        except IOError:
            continue

        if len(data) == 0:
            continue

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text   = result.get("text", "").lower().strip()

            if text:
                process_text_command(text, source_type="Voice")

# ── Zero-Resource SSH Optional Listener Thread ────────
def optional_ssh_thread():
    """
    Blocks cleanly on the OS file system layer using a Linux Pipe.
    Consumes 0% CPU and 0 MB RAM when no one is using SSH.
    """
    print("🖥️ Background SSH Monitor Initialized (Zero-Overhead).")
    
    while True:
        # Opening a named pipe blocks execution until data enters the channel
        with open(PIPE_PATH, "r") as fifo:
            for line in fifo:
                clean_command = line.strip().lower()
                if clean_command:
                    # Pass incoming terminal inputs directly to screen layout execution
                    process_text_command(clean_command, source_type="SSH")

# ── Draw overlay on frame ──────────────────────────────
def draw_overlay(frame, faces, heard, response, status):
    h, w = frame.shape[:2]

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Status box background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if heard:
        cv2.putText(frame, heard, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    if response:
        cv2.putText(frame, response, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, status, (w - 180, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    face_label = f"Faces: {len(faces)}"
    cv2.putText(frame, face_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

    return frame

def execute_command(action):
    print(f"Motor Action Triggered: {action}")

# ── Camera + display (main thread) ────────────────────
def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("AURA", cv2.WINDOW_NORMAL)
    cv2.moveWindow("AURA", 0, 0)
    cv2.resizeWindow("AURA", DISPLAY_W, DISPLAY_H)
    cv2.setWindowProperty("AURA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Start Voice Listener
    t_voice = threading.Thread(target=voice_thread, daemon=True)
    t_voice.start()

    # Start Zero-Resource SSH Listener
    t_ssh = threading.Thread(target=optional_ssh_thread, daemon=True)
    t_ssh.start()

    with state_lock:
        state["status"] = "AURA Active"

    frame_count = 0
    local_faces = [] 

    while True:
        frame = picam2.capture_array()
     

        if frame_count % 5 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_boxes = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3,
                minNeighbors=5, minSize=(30, 30)
            )
            local_faces = [tuple(b) for b in detected_boxes] if len(detected_boxes) > 0 else []
            
            with state_lock:
                state["faces"] = local_faces

            handle_greeting(len(local_faces))
            frame_count = 0
        frame_count += 1

        with state_lock:
            heard    = state["heard_text"]
            response = state["response_text"]
            status   = state["status"]

        frame = draw_overlay(frame, local_faces, heard, response, status)

        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("AURA", display_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()

