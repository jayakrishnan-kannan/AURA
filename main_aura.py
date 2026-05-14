import pyaudio  
import sys
import select
from picamera2 import Picamera2
import cv2
import threading
import time
import os
import json
import subprocess 
from vosk import Model, KaldiRecognizer
from groq import Groq  

# ── Constants & API Configuration ─────────────────────
DISPLAY_W, DISPLAY_H = 800, 480
CASCADE_PATH = "/home/aura/AURA/model/haarcascade_frontalface_default.xml"
MODEL_PATH   = "/home/aura/AURA/model/vosk-model-small-en-in-0.4"
PIPE_PATH    = "/home/aura/AURA/ssh_bridge.pipe"

# Paste your actual key as the variable value here
GROQ_API_KEY = ""


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
    "display_active": False,    # 👈 CRITICAL FIX: Locks HUD visibility at frame level
    "face_lost_time": None,    
}
state_lock = threading.Lock()

# ── Synchronous Hard-Blocking TTS Engine ───────────────
def speak_blocking(text):
    try:
        p1 = subprocess.Popen(["espeak", "-s", "160", text, "--stdout"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p2 = subprocess.Popen(["aplay"], stdin=p1.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p1.stdout.close()
        p2.communicate()  
    except Exception as e:
        print(f"Audio Output Error: {e}")

# ── Asynchronous Background TTS (For Greetings) ────────
def speak(text):
    def _speak_task():
        with state_lock:
            state["speaking"] = True
            state["display_active"] = True
        speak_blocking(text)
        with state_lock:
            state["speaking"] = False
            state["display_active"] = False
            state["heard_text"] = ""
            state["response_text"] = ""
    threading.Thread(target=_speak_task, daemon=True).start()

# ── Free Groq Cloud Gateway ───────────────────────────
def ask_ai_brain(question):
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return "API Token layer missing."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are AURA, a smart visual robot assistant. Give short answers under 12 words."},
                {"role": "user", "content": question}
            ],
            timeout=4.0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "Brain network timeout."

# ── Centralized Text Display Engine ────────────────────
def process_text_command(text_input, source_type="Input"):
    if not text_input:
        return

    # 1. Open the HUD and project the text immediately
    with state_lock:
        state["heard_text"] = f"{source_type}: {text_input}"
        state["response_text"] = "Thinking..."
        state["display_active"] = True

    matched = None
    for keyword, action in COMMANDS.items():
        if keyword in text_input:
            matched = action
            break

    if matched:
        response = matched.replace("_", " ")
        with state_lock:
            state["response_text"] = f"CMD: {response}"
            state["speaking"] = True
        
        # Hold state lock during audio execution
        speak_blocking(response)
        
        with state_lock:
            state["speaking"] = False
            state["display_active"] = False  # 👈 Clear screen ONLY after voice finishes
            state["heard_text"] = ""
            state["response_text"] = ""
            
        execute_command(matched)
    else:
        ai_response = ask_ai_brain(text_input)
        
        with state_lock:
            state["response_text"] = ai_response
            state["speaking"] = True  
        
        # Enforce minimum display duration safety timer
        start_time = time.time()
        speak_blocking(ai_response)
        elapsed = time.time() - start_time
        
        if elapsed < 2.5:
            time.sleep(2.5 - elapsed)
        
        with state_lock:
            state["speaking"] = False
            state["display_active"] = False  # 👈 Clear screen ONLY after voice finishes
            state["heard_text"] = ""
            state["response_text"] = ""

def handle_greeting(face_count):
    with state_lock:
        greeted    = state["greeted"]
        speaking   = state["speaking"]  

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
                if not state["display_active"]:
                    state["response_text"] = ""

# ── Input Streams Thread Pools ────────────────────────
def voice_thread():
    time.sleep(2)  
    try:
        model = Model(MODEL_PATH)
        rec   = KaldiRecognizer(model, 16000)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    except Exception as e:
        with state_lock: state["status"] = "SSH Only Mode"
        while True: time.sleep(1)

    with state_lock: state["status"] = "AURA Listening"
    while True:
        try: data = stream.read(4000, exception_on_overflow=False)
        except IOError: continue
        if len(data) == 0: continue
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text   = result.get("text", "").lower().strip()
            if text: process_text_command(text, source_type="Voice")

def optional_ssh_thread():
    while True:
        with open(PIPE_PATH, "r") as fifo:
            for line in fifo:
                clean_command = line.strip().lower()
                if clean_command: process_text_command(clean_command, source_type="SSH")

# ── Micro-HUD Display Overlay Engine (1/6th Height) ────
def draw_multiline_text(frame, text, start_x, start_y, max_width, line_height, font, scale, color, thickness):
    words = text.split(' ')
    current_line = ""
    y = start_y

    for word in words:
        test_line = current_line + word + " "
        (line_w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
        
        if line_w > max_width and current_line:
            cv2.putText(frame, current_line.strip(), (start_x, y), font, scale, color, thickness, cv2.LINE_AA)
            current_line = word + " "
            y += line_height  
        else:
            current_line = test_line

    if current_line:
        cv2.putText(frame, current_line.strip(), (start_x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_overlay(frame, faces, heard, response, status, display_active):
    h, w = frame.shape[:2]

    # Target Tracking Boxes
    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        cv2.putText(frame, "Target", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # 👈 FIX: Evaluate display_active barrier state explicitly
    if display_active or heard or response:
        hud_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.48      
        text_thick = 1        

        if heard:
            cv2.putText(frame, heard, (15, h - 60), font, text_scale, (0, 255, 255), text_thick, cv2.LINE_AA)

        if response:
            draw_multiline_text(
                frame=frame, text=response, start_x=15, start_y=h - 38,
                max_width=770, line_height=18, font=font,
                scale=text_scale, color=(0, 255, 0), thickness=text_thick + 1
            )

    # Top Status Text Headers
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, status, (w - 150, 20), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Trackers: {len(faces)}", (15, 20), font, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    return frame

def execute_command(action):
    print(f"Motor Action Triggered: {action}")

def main():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("AURA", cv2.WINDOW_NORMAL)
    cv2.moveWindow("AURA", 0, 0)
    cv2.resizeWindow("AURA", DISPLAY_W, DISPLAY_H)
    cv2.setWindowProperty("AURA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    threading.Thread(target=voice_thread, daemon=True).start()
    threading.Thread(target=optional_ssh_thread, daemon=True).start()

    with state_lock: state["status"] = "System Online"
    frame_count = 0
    local_faces = [] 

    while True:
        frame = picam2.capture_array()

        if frame_count % 5 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            local_faces = [tuple(b) for b in detected_boxes] if len(detected_boxes) > 0 else []
            with state_lock: state["faces"] = local_faces
            handle_greeting(len(local_faces))
            frame_count = 0
        frame_count += 1

        # Atomically snap all layout states together
        with state_lock:
            heard          = state["heard_text"]
            response       = state["response_text"]
            status         = state["status"]
            display_active = state["display_active"] # 👈 NEW: Synchronized flag map

        frame = draw_overlay(frame, local_faces, heard, response, status, display_active)
        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("AURA", display_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()

