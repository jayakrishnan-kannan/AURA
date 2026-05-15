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
import RPi.GPIO as GPIO

# ── Constants & API Configuration ─────────────────────
DISPLAY_W, DISPLAY_H = 800, 480
CASCADE_PATH = "/home/aura/AURA/model/haarcascade_frontalface_default.xml"
MODEL_PATH = "/home/aura/AURA/model/vosk-model-small-en-in-0.4"
PIPE_PATH = "/home/aura/AURA/ssh_bridge.pipe"

GROQ_API_KEY = ""

# ── Motor GPIO Pin Map (BCM numbering) ─────────────────
MOTOR_PINS = {
    "left_front": {"in1": 5, "in2": 6},
    "left_rear": {"in1": 12, "in2": 13},
    "right_front": {"in1": 19, "in2": 16},
    "right_rear": {"in1": 26, "in2": 20},
}

# ── Voice / SSH Command Map ────────────────────────────
COMMANDS = {
    "forward": "move_forward",
    "back": "move_backward",
    "left": "turn_left",
    "right": "turn_right",
    "stop": "stop_motors",
}

# ── Shared state ───────────────────────────────────────
state = {
    "faces": [],
    "heard_text": "",
    "response_text": "",
    "status": "Initializing...",
    "greeted": False,
    "speaking": False,
    "display_active": False,
    "face_lost_time": None,
    "scroll_offset": 0,  # Tracks lines shifting upwards for long responses
}
state_lock = threading.Lock()

# ══════════════════════════════════════════════════════
# GPIO MOTOR CONTROL
# ══════════════════════════════════════════════════════


def gpio_setup():
    """
    Initialize all motor GPIO pins. Called once at main startup loop.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for motor, pins in MOTOR_PINS.items():
        GPIO.setup(pins["in1"], GPIO.OUT)
        GPIO.setup(pins["in2"], GPIO.OUT)
    _all_stop()
    print("⚙️ GPIO Motor Pins Successfully Initialised (BCM Mode Registered).")


def gpio_cleanup():
    """
    Safe cleanup — stops all motors and releases GPIO configurations on exit.
    """
    _all_stop()
    GPIO.cleanup()
    print("🧹 GPIO Cleaned Up Safe and Released.")


def _set_motor(motor, direction):
    pins = MOTOR_PINS[motor]
    if direction == "fwd":
        GPIO.output(pins["in1"], GPIO.HIGH)
        GPIO.output(pins["in2"], GPIO.LOW)
    elif direction == "bwd":
        GPIO.output(pins["in1"], GPIO.LOW)
        GPIO.output(pins["in2"], GPIO.HIGH)
    else:  # stop
        GPIO.output(pins["in1"], GPIO.LOW)
        GPIO.output(pins["in2"], GPIO.LOW)


def _all_stop():
    for motor in MOTOR_PINS:
        _set_motor(motor, "stop")


def _drive(left, right):
    _set_motor("left_front", left)
    _set_motor("left_rear", left)
    _set_motor("right_front", right)
    _set_motor("right_rear", right)


def execute_command(action):
    print(f"⚡ Executing Hardware Motor Action: {action}")
    if action == "move_forward":
        _drive("fwd", "fwd")
    elif action == "move_backward":
        _drive("bwd", "bwd")
    elif action == "turn_left":
        _drive("bwd", "fwd")
    elif action == "turn_right":
        _drive("fwd", "bwd")
    elif action == "stop_motors":
        _all_stop()
    else:
        print(f"Unknown action: {action}")


# ══════════════════════════════════════════════════════
# TTS ENGINE
# ══════════════════════════════════════════════════════


def speak_blocking(text):
    """
    Synchronous audio stream holder. Keeps text printed on screen until
    audio output hardware completely completes the sentence string block.
    """
    try:
        p1 = subprocess.Popen(
            ["espeak", "-s", "160", text, "--stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        p2 = subprocess.Popen(
            ["aplay"],
            stdin=p1.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        p1.stdout.close()
        p2.communicate()
    except Exception as e:
        print(f"Audio Output Error: {e}")


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
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are AURA, a smart visual robot assistant. Give short answers under 15 words.",
                },
                {"role": "user", "content": question},
            ],
            timeout=4.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Actual Error Stack Trace Info: {e}")
        return "Brain network timeout."


# ── Centralized Text Display Engine ────────────────────
def process_text_command(text_input, source_type="Input"):
    if not text_input:
        return

    with state_lock:
        state["heard_text"] = f"{source_type}: {text_input}"
        state["response_text"] = "Thinking..."
        state["display_active"] = True
        state["scroll_offset"] = 0

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

        speak_blocking(response)
        execute_command(matched)

        with state_lock:
            state["speaking"] = False
            state["display_active"] = False
            state["heard_text"] = ""
            state["response_text"] = ""
    else:
        ai_response = ask_ai_brain(text_input)

        with state_lock:
            state["response_text"] = ai_response
            state["speaking"] = True

        # Slices paragraph string into distinct sentence pieces to sync with scroll engine pages
        sub_sentences = ai_response.replace("?", ".").split(".")
        sub_sentences = [s.strip() for s in sub_sentences if s.strip()]

        if len(sub_sentences) > 1:
            for i, sentence in enumerate(sub_sentences):
                with state_lock:
                    state["scroll_offset"] = (
                        i * 1
                    )  # Shift layout lines up frame by frame
                speak_blocking(sentence)
        else:
            speak_blocking(ai_response)
            time.sleep(1.5)

        with state_lock:
            state["speaking"] = False
            state["display_active"] = False
            state["heard_text"] = ""
            state["response_text"] = ""


def handle_greeting(face_count):
    with state_lock:
        greeted = state["greeted"]
        speaking = state["speaking"]

    if face_count > 0:
        with state_lock:
            state["face_lost_time"] = None
        if not greeted and not speaking:
            with state_lock:
                state["greeted"] = True
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
                state["greeted"] = False
                if not state["display_active"]:
                    state["response_text"] = ""


# ── Input Streams Thread Pools ────────────────────────
def voice_thread():
    time.sleep(2)
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    use_hardware_mic = True
    stream = None
    fifo_file = None

    try:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4000,
        )
        with state_lock:
            state["status"] = "AURA Mic Active"
        print("Hardware Audio Input bound.")
    except Exception as e:
        use_hardware_mic = False
        with state_lock:
            state["status"] = "AURA Sim Mic"
        print(f"Hardware mic not found ({e}). Connecting to virtual FIFO...")
        try:
            fifo_file = open("/home/aura/AURA/mic_sim.pipe", "rb")
            print("Virtual Audio Pipe linked.")
        except Exception as pipe_err:
            print(f"Pipe failed: {pipe_err}")
            with state_lock:
                state["status"] = "Audio Error"
            return

    while True:
        try:
            if use_hardware_mic:
                data = stream.read(4000, exception_on_overflow=False)
            else:
                data = fifo_file.read(8000)
                if len(data) == 0:
                    time.sleep(0.01)
                    continue
        except IOError:
            continue

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "").lower().strip()
            if text:
                process_text_command(text, source_type="Voice")


def optional_ssh_thread():
    while True:
        with open(PIPE_PATH, "r") as fifo:
            for line in fifo:
                clean_command = line.strip().lower()
                if clean_command:
                    process_text_command(clean_command, source_type="SSH")


# ── Micro-HUD Display Overlay Engine with Auto-Scaling ──
def draw_scaled_multiline_text(
    frame,
    text,
    start_x,
    start_y,
    max_width,
    base_line_height,
    font,
    thickness,
    scroll_offset,
):
    # Font contraction metrics configuration based on text density
    if len(text) > 60:
        scale = 0.36
        line_height = 14
    else:
        scale = 0.46
        line_height = base_line_height

    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        (line_w, _), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if line_w > max_width and current_line:
            lines.append(current_line.strip())
            current_line = word + " "
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line.strip())

    # Crop and index lines array based on scroll_offset paged instructions
    visible_lines = lines[scroll_offset:]
    y = start_y
    for i, line in enumerate(visible_lines):
        if i >= 2:  # Bound to maximum 2 rows inside the 80 pixel boundary
            break
        cv2.putTextBox = cv2.putText(
            frame, line, (start_x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA
        )
        y += line_height


def draw_overlay(frame, faces, heard, response, status, display_active, scroll_offset):
    h, w = frame.shape[:2]

    # Target Tracking Boxes
    for x, y, fw, fh in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Target",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Dynamic 1/6th HUD panel renderer
    if display_active or heard or response:
        hud_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if heard:
            cv2.putText(
                frame, heard, (15, h - 62), font, 0.44, (0, 255, 255), 1, cv2.LINE_AA
            )

        if response:
            draw_scaled_multiline_text(
                frame=frame,
                text=response,
                start_x=15,
                start_y=h - 40,
                max_width=770,
                base_line_height=18,
                font=font,
                thickness=1,
                scroll_offset=scroll_offset,
            )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame, status, (w - 150, 20), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Trackers: {len(faces)}",
        (15, 20),
        font,
        0.42,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame


# ── Main Application Engine ───────────────────────────
def main():
    # 👈 FIX: Call the initialization line immediately inside main loop startup!
    gpio_setup()

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

    threading.Thread(target=voice_thread, daemon=True).start()
    threading.Thread(target=optional_ssh_thread, daemon=True).start()

    with state_lock:
        state["status"] = "System Online"

    frame_count = 0
    local_faces = []

    try:
        while True:
            frame = picam2.capture_array()

            if frame_count % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_boxes = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                )
                local_faces = (
                    [tuple(b) for b in detected_boxes]
                    if len(detected_boxes) > 0
                    else []
                )
                with state_lock:
                    state["faces"] = local_faces
                handle_greeting(len(local_faces))
                frame_count = 0
            frame_count += 1

            with state_lock:
                heard = state["heard_text"]
                response = state["response_text"]
                status = state["status"]
                display_active = state["display_active"]
                scroll_offset = state["scroll_offset"]

            frame = draw_overlay(
                frame,
                local_faces,
                heard,
                response,
                status,
                display_active,
                scroll_offset,
            )
            display_frame = cv2.resize(
                frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR
            )
            cv2.imshow("AURA", display_frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    finally:
        # Secure lifecycle termination guardrail execution
        gpio_cleanup()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
