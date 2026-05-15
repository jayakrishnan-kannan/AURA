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

# Production Groq API Key
GROQ_API_KEY = ""

# ── Motor GPIO Pin Map (BCM Numbering) ─────────────────
MOTOR_PINS = {
    "left_front": {"in1": 5, "in2": 6},
    "left_rear": {"in1": 12, "in2": 13},
    "right_front": {"in1": 19, "in2": 16},
    "right_rear": {"in1": 26, "in2": 20},
}

# ── Ultrasonic Sensor GPIO Pin Map (BCM Numbering) ─────
TRIG_PIN = 27
ECHO_PIN = 17
OBSTACLE_THRESHOLD_CM = 20.0  # Stop distance threshold

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
    "scroll_offset": 0,
}
state_lock = threading.Lock()

# ── Globals for Motor Tracking Across Threads ──────────
pwm_objects = {}
current_speed = 0
current_dir_left = "stop"
current_dir_right = "stop"
ramp_lock = threading.Lock()

# ── Ultrasonic Watchdog State Trackers ────────────────
last_intended_command = "stop_motors"
obstacle_blocked = False

# ══════════════════════════════════════════════════════
# ADVANCED POWER-MANAGED MOTOR CONTROL (PWM RAMPING)
# ══════════════════════════════════════════════════════


def gpio_setup():
    """
    Initialises physical pins and attaches PWM frequencies at a silent 1kHz.
    Called once inside the main() startup block.
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    global pwm_objects
    for motor, pins in MOTOR_PINS.items():
        GPIO.setup(pins["in1"], GPIO.OUT)
        GPIO.setup(pins["in2"], GPIO.OUT)

        # Spawn isolated PWM channels per pin input lane
        pwm_objects[f"{motor}_in1"] = GPIO.PWM(pins["in1"], 1000)
        pwm_objects[f"{motor}_in2"] = GPIO.PWM(pins["in2"], 1000)

        pwm_objects[f"{motor}_in1"].start(0)
        pwm_objects[f"{motor}_in2"].start(0)

    # Setup Ultrasonic Pins
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    print("⚙️ PWM Matrix & Ultrasonic Sensor Initialised Successfully.")


def gpio_cleanup():
    """
    Safe cleanup — stops all motors and releases GPIO configurations on exit.
    """
    global current_speed, current_dir_left, current_dir_right
    current_speed = 0
    current_dir_left = "stop"
    current_dir_right = "stop"
    _update_all_wheels(0)
    GPIO.cleanup()
    print("🧹 GPIO Cleaned Up Safe and Released.")


def _apply_hardware_pwm(motor, direction, speed_pct):
    """
    Direct low-level hardware voltage register writer.
    """
    p_in1 = pwm_objects[f"{motor}_in1"]
    p_in2 = pwm_objects[f"{motor}_in2"]

    if direction == "fwd":
        p_in1.ChangeDutyCycle(speed_pct)
        p_in2.ChangeDutyCycle(0)
    elif direction == "bwd":
        p_in1.ChangeDutyCycle(0)
        p_in2.ChangeDutyCycle(speed_pct)
    else:  # stop
        p_in1.ChangeDutyCycle(0)
        p_in2.ChangeDutyCycle(0)


def _update_all_wheels(speed_pct):
    """
    Applies the matching intermediate speed to all wheels simultaneously
    based on the current structural direction states.
    """
    global current_dir_left, current_dir_right
    _apply_hardware_pwm("left_front", current_dir_left, speed_pct)
    _apply_hardware_pwm("left_rear", current_dir_left, speed_pct)
    _apply_hardware_pwm("right_front", current_dir_right, speed_pct)
    _apply_hardware_pwm("right_rear", current_dir_right, speed_pct)


def manage_motion_sequence(
    target_speed, target_left_dir, target_right_dir, duration=0.4
):
    """
    1. If the robot is moving, it first decelerates down to 0% cleanly.
    2. Once safely at zero, it shifts the directional switches (no plugging).
    3. It then accelerates smoothly up to your target velocity speed.
    """
    global current_speed, current_dir_left, current_dir_right

    with ramp_lock:
        # Step A: Safe Deceleration Phase (If currently moving)
        if current_speed > 0:
            steps = 8
            sleep_time = 0.2 / steps
            while current_speed > 0:
                current_speed -= (
                    (current_speed / steps) if current_speed > 5 else current_speed
                )
                current_speed = max(0, current_speed)
                _update_all_wheels(int(current_speed))
                time.sleep(sleep_time)

            # Absolute mechanical rest pause to let motor coils settle
            current_dir_left = "stop"
            current_dir_right = "stop"
            _update_all_wheels(0)
            time.sleep(0.1)

        # Step B: Shift Polarity Safely at Zero Velocity
        current_dir_left = target_left_dir
        current_dir_right = target_right_dir

        # Step C: Smooth Acceleration Phase up to Target Power
        if target_speed > 0:
            steps = 10
            sleep_time = duration / steps
            step_increment = target_speed / steps
            while current_speed < target_speed:
                current_speed += step_increment
                current_speed = min(target_speed, current_speed)
                _update_all_wheels(int(current_speed))
                time.sleep(sleep_time)


def execute_command(action):
    """
    Threaded execution interface maps voice/SSH prompts to hardware pipelines.
    Spawns background tasks so movement ramping doesn't stall OpenCV display frames.
    """
    global last_intended_command, obstacle_blocked

    if action != "stop_motors":
        last_intended_command = action

    # If an obstacle is blocking the path, intercept and refuse forward movements
    if obstacle_blocked and action == "move_forward":
        print("🛑 Obstacle Block Active: Refusing forward request.")
        return

    print(f"⚡ Processing Managed Power Action: {action}")
    if action == "move_forward":
        threading.Thread(
            target=manage_motion_sequence, args=(80, "fwd", "fwd"), daemon=True
        ).start()
    elif action == "move_backward":
        threading.Thread(
            target=manage_motion_sequence, args=(80, "bwd", "bwd"), daemon=True
        ).start()
    elif action == "turn_left":
        threading.Thread(
            target=manage_motion_sequence, args=(65, "bwd", "fwd"), daemon=True
        ).start()
    elif action == "turn_right":
        threading.Thread(
            target=manage_motion_sequence, args=(65, "fwd", "bwd"), daemon=True
        ).start()
    elif action == "stop_motors":
        threading.Thread(
            target=manage_motion_sequence, args=(0, "stop", "stop"), daemon=True
        ).start()


# ══════════════════════════════════════════════════════
# ULTRASONIC CRUISE WATCHDOG THREAD (ZERO CPU OVERHEAD)
# ══════════════════════════════════════════════════════


def measure_distance():
    """
    Calculates physical object proximity using high-precision hardware timestamps.
    """
    try:
        GPIO.output(TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)  # 10 microsecond trigger pulse
        GPIO.output(TRIG_PIN, GPIO.LOW)

        pulse_start = time.time()
        pulse_end = time.time()

        # Timeout counters prevent hanging if the echo pulse is missed
        timeout = time.time()
        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()
            if pulse_start - timeout > 0.05:
                return 999.0

        timeout = time.time()
        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()
            if pulse_end - timeout > 0.05:
                return 999.0

        pulse_duration = pulse_end - pulse_start
        distance = (
            pulse_duration * 17150
        )  # Distance calculation based on speed of sound
        return round(distance, 1)
    except Exception:
        return 999.0


def ultrasonic_watchdog_thread():
    """
    Monitors obstacles continuously. Halts forward execution if blocked,
    and automatically resumes movement when the path clears.
    """
    global obstacle_blocked, last_intended_command
    time.sleep(2.5)  # Wait for display server to boot

    print("🛰️ Ultrasonic Range Watchdog Active.")

    while True:
        dist = measure_distance()

        if dist < OBSTACLE_THRESHOLD_CM:
            if not obstacle_blocked:
                # Obstacle detected for the first time
                obstacle_blocked = True
                with state_lock:
                    state["status"] = "⚠️ OBSTACLE DETECTED"

                # Halt forward movement if active
                if last_intended_command == "move_forward":
                    print("🚨 Proximity Alert! Initiating emergency auto-stop.")
                    threading.Thread(
                        target=manage_motion_sequence,
                        args=(0, "stop", "stop"),
                        daemon=True,
                    ).start()
        else:
            if obstacle_blocked:
                # Obstacle removed, path is clear
                obstacle_blocked = False
                with state_lock:
                    state["status"] = "System Online"
                print("✅ Path Clear. Reviewing auto-resume logs.")

                # Resume moving forward if that was the last intended command
                if last_intended_command == "move_forward":
                    print("🚀 Auto-Resuming forward cruise sequence.")
                    threading.Thread(
                        target=manage_motion_sequence,
                        args=(80, "fwd", "fwd"),
                        daemon=True,
                    ).start()

        time.sleep(0.1)  # 10Hz polling rate limits CPU overhead


# ══════════════════════════════════════════════════════
# SYNCHRONOUS TTS ENGINE
# ══════════════════════════════════════════════════════


def speak_blocking(text):
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
        return completion.choices.message.content.strip()
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
                    state["scroll_offset"] = i * 1
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
    """
    Dual Audio Stream Pipeline.
    1. Attempts to open physical USB Microphone via PyAudio.
    2. Fallback: Opens a Linux Named Pipe to receive raw binary testing audio.
    """
    time.sleep(2)  # Allow camera and UI engine to initialize first

    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)

    use_hardware_mic = True
    stream = None
    fifo_file = None

    # 🎙️ Try connecting to physical hardware microphone first
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
        print("🎙️ Live Hardware Microphone stream initialized successfully.")
    except Exception as e:
        # 🧪 FALLBACK: Trigger file pipe channel if mic is absent
        use_hardware_mic = False
        with state_lock:
            state["status"] = "AURA Sim Mic"
        print(
            f"⚠️ Mic missing ({e}). Activating Sim Pipe: /home/aura/AURA/mic_sim.pipe"
        )

        # Open the simulation audio named pipe in Read-Binary mode
        try:
            fifo_file = open("/home/aura/AURA/mic_sim.pipe", "rb")
        except Exception as pipe_err:
            print(f"❌ Failed to open mic sim pipe: {pipe_err}")
            return

    # Main speech parsing engine loop
    while True:
        try:
            if use_hardware_mic:
                # Read chunks directly from the physical USB driver chip
                data = stream.read(4000, exception_on_overflow=False)
            else:
                # Read chunks directly from your simulated terminal audio injector
                data = fifo_file.read(8000)
                if len(data) == 0:
                    time.sleep(0.01)  # Stop thread from pinning CPU at 100% when idle
                    continue
        except IOError:
            continue

        # Feed extracted data frames straight into the Vosk text engine
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

    visible_lines = lines[scroll_offset:]
    y = start_y
    for i, line in enumerate(visible_lines):
        if i >= 2:
            break
        cv2.putText(
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
    # Call the initialization line immediately inside main loop startup
    gpio_setup()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (320, 240), "format": "BGR888"}
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
