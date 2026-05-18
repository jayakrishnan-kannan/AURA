import pyaudio
from picamera2 import Picamera2
import cv2
import threading
import time
import socket
import json
import subprocess
import logging
from vosk import Model, KaldiRecognizer
from groq import Groq
import RPi.GPIO as GPIO
from flask import Flask, Response, render_template_string, request

# ── Suppress Annoying 404/Muted HTTP Logs ───────────────
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# ── Constants & API Configuration ─────────────────────
DISPLAY_W, DISPLAY_H = 800, 480
CASCADE_PATH = "/home/aura/AURA/model/haarcascade_frontalface_default.xml"
MODEL_PATH = "/home/aura/AURA/model/vosk-model-small-en-in-0.4"
PIPE_PATH = "/home/aura/AURA/ssh_bridge.pipe"

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
OBSTACLE_THRESHOLD_CM = 20.0

# ── Voice / SSH Command Map ────────────────────────────
COMMANDS = {
    "forward": "move_forward",
    "back": "move_backward",
    "left": "turn_left",
    "right": "turn_right",
    "stop": "stop_motors",
}

# ── Full JEC & ECE Local Custom Response Cache Matrix ──
LOCAL_QA_CACHE = {
    # --- AURA General Persona Questions ---
    "name": "Hello, I am AURA, Autonomous Unified Robotic Assistant.",
    "hello": "Hello, how can I assist you?",
    "hi": "Hi, nice to meet you.",
    "good morning": "Good morning. How can I help you today?",
    "good evening": "Good evening. Hope you are doing well.",
    "what can you do": "I can provide information, assist visitors, and guide users inside the campus.",
    "features": "My features include voice interaction, face detection, robotic movement, and smart assistance.",
    "purpose": "I am designed to assist users through voice interaction and robotic automation.",
    "detect faces": "Yes, I can detect faces using my camera module and OpenCV.",
    "move": "Yes, I can move based on commands.",
    "thank you": "You are welcome.",
    "goodbye": "Goodbye. Have a nice day.",
    "ready": "Hello, I am AURA. How can I assist you today?",
    # --- Hardware Motor Movements ---
    "move forward": "Moving forward.",
    "move backward": "Moving backward.",
    # --- Jeppiaar Engineering College (JEC) ---
    "about jec": "Jeppiaar Engineering College is one of the leading autonomous engineering colleges in Chennai providing quality education, placement support, research opportunities, and modern infrastructure.",
    "tell me about jec": "Jeppiaar Engineering College is one of the leading autonomous engineering colleges in Chennai providing quality education, placement support, research opportunities, and modern infrastructure.",
    "founded jec": "Colonel Dr. Jeppiaar founded Jeppiaar Engineering College, and now Chancellor Regena J Murali Mam is taking the institution to the next level through innovation, academic excellence, and student development.",
    "who founded": "Colonel Dr. Jeppiaar founded Jeppiaar Engineering College, and now Chancellor Regena J Murali Mam is taking the institution to the next level through innovation, academic excellence, and student development.",
    "chancellor": "Regena J Murali Mam is the Chancellor of Jeppiaar Engineering College.",
    "autonomous": "Yes, JEC is an autonomous institution.",
    "located": "JEC is located in Semmancheri, Chennai.",
    "courses": "JEC offers courses including CSE, AI and DS, AI and ML, IT, ECE, EEE, Mechanical, Civil, and Biotechnology.",
    "placement": "Yes, the college provides placement assistance and career training.",
    "hostel": "Yes, hostel facilities are available for students staying in campus.",
    "hostile": "Yes, hostel facilities are available for students staying in campus.",  # Speech fix
    "transport": "Yes, transportation facilities are available for day scholars.",
    "food": "Yes, food facilities are available for both hostel students and day scholars.",
    # --- ECE Department ---
    "ece department": "The ECE department of Jeppiaar Engineering College is one of the best departments in the college, known for quality education, experienced faculty, advanced laboratories, innovative projects, and accredited programs in electronics, communication, networking, IoT, and robotics.",
    "accredited": "Yes, the ECE department is accredited and provides quality technical education.",
    "hod cabin": "The HOD cabin is on the first floor of the ECE department.",
    "staff room": "The ECE staff room is on the second floor.",
    # --- Directions ---
    "library": "The library is located in the blue building.",
    "auditorium": "The auditorium is on the first floor of the blue building.",
    "biotechnology": "The biotechnology department is beside the blue building on the right side.",
    "bio technology": "The biotechnology department is beside the blue building on the right side.",  # Speech fix
    "mess": "The mess is located on the left side of the blue building.",
    "it department": "The IT department is opposite to the ECE department.",
    "ai and ds": "The AI and DS department is beside the ECE department on the left side.",
    "cse department": "The CSE department is beside the ECE department.",
    "placement cell": "The placement cell is located behind the ECE department.",
    "admission office": "The admission office is located near the entrance on the right side of ECE.",
    "canteen": "The canteen is located near the academic block.",
    "seminar hall": "The seminar hall is located in the blue building.",
}

# ── Shared State Architecture ─────────────────────────
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

# ── 👈 NEW: Flask Web Server Frame Buffer ──────────────
app = Flask(__name__)
global_web_frame = None
frame_buffer_lock = threading.Lock()

# ══════════════════════════════════════════════════════
# MOTOR CONTROL AND ULTRASONIC HARDWARE
# ══════════════════════════════════════════════════════


def gpio_setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    global pwm_objects
    for motor, pins in MOTOR_PINS.items():
        GPIO.setup(pins["in1"], GPIO.OUT)
        GPIO.setup(pins["in2"], GPIO.OUT)
        pwm_objects[f"{motor}_in1"] = GPIO.PWM(pins["in1"], 1000)
        pwm_objects[f"{motor}_in2"] = GPIO.PWM(pins["in2"], 1000)
        pwm_objects[f"{motor}_in1"].start(0)
        pwm_objects[f"{motor}_in2"].start(0)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, GPIO.LOW)
    print("⚙️ PWM Matrix & Ultrasonic Sensor Initialised Successfully.")


def gpio_cleanup():
    global current_speed, current_dir_left, current_dir_right
    current_speed = 0
    current_dir_left = "stop"
    current_dir_right = "stop"
    _update_all_wheels(0)
    GPIO.cleanup()
    print("🧹 GPIO Cleaned Up Safe and Released.")


def _apply_hardware_pwm(motor, direction, speed_pct):
    p_in1 = pwm_objects[f"{motor}_in1"]
    p_in2 = pwm_objects[f"{motor}_in2"]
    if direction == "fwd":
        p_in1.ChangeDutyCycle(speed_pct)
        p_in2.ChangeDutyCycle(0)
    elif direction == "bwd":
        p_in1.ChangeDutyCycle(0)
        p_in2.ChangeDutyCycle(speed_pct)
    else:
        p_in1.ChangeDutyCycle(0)
        p_in2.ChangeDutyCycle(0)


def _update_all_wheels(speed_pct):
    global current_dir_left, current_dir_right
    _apply_hardware_pwm("left_front", current_dir_left, speed_pct)
    _apply_hardware_pwm("left_rear", current_dir_left, speed_pct)
    _apply_hardware_pwm("right_front", current_dir_right, speed_pct)
    _apply_hardware_pwm("right_rear", current_dir_right, speed_pct)


def manage_motion_sequence(
    target_speed, target_left_dir, target_right_dir, duration=0.4
):
    global current_speed, current_dir_left, current_dir_right
    with ramp_lock:
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
            current_dir_left = "stop"
            current_dir_right = "stop"
            _update_all_wheels(0)
            time.sleep(0.1)

        current_dir_left = target_left_dir
        current_dir_right = target_right_dir

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
    global last_intended_command, obstacle_blocked
    if action != "stop_motors":
        last_intended_command = action
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


def measure_distance():
    try:
        GPIO.output(TRIG_PIN, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, GPIO.LOW)
        pulse_start = time.time()
        pulse_end = time.time()
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
        return round((pulse_end - pulse_start) * 17150, 1)
    except Exception:
        return 999.0


def ultrasonic_watchdog_thread():
    global obstacle_blocked, last_intended_command
    time.sleep(2.5)
    print("🛰️ Ultrasonic Range Watchdog Active.")
    while True:
        dist = measure_distance()
        if dist < OBSTACLE_THRESHOLD_CM:
            if not obstacle_blocked:
                obstacle_blocked = True
                with state_lock:
                    state["status"] = "⚠️ OBSTACLE"
                if last_intended_command == "move_forward":
                    print("🚨 Proximity Alert! Initiating emergency auto-stop.")
                    threading.Thread(
                        target=manage_motion_sequence,
                        args=(0, "stop", "stop"),
                        daemon=True,
                    ).start()
        else:
            if obstacle_blocked:
                obstacle_blocked = False
                with state_lock:
                    state["status"] = "System Online"
                print("✅ Path Clear. Reviewing auto-resume logs.")
                if last_intended_command == "move_forward":
                    print("🚀 Auto-Resuming forward cruise sequence.")
                    threading.Thread(
                        target=manage_motion_sequence,
                        args=(80, "fwd", "fwd"),
                        daemon=True,
                    ).start()
        time.sleep(0.1)


# ══════════════════════════════════════════════════════
# AUDIO SYNTHESIS & AI COGNITION CORE
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


def ask_ai_brain(question):
    clean_question = question.strip().lower()
    for key, local_answer in LOCAL_QA_CACHE.items():
        if key in clean_question:
            print(
                f'📦 [LOCAL CACHE MATCH ACQUIRED]: Found keyword token "{key}" offline.'
            )
            return local_answer
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


def process_text_command(text_input, source_type="Input"):
    if not text_input:
        return
    print(f'\n📥 [{source_type} QUERY RECEIVED]: "{text_input}"')
    with state_lock:
        state["heard_text"] = f"{source_type}: {text_input}"
        state["response_text"] = "Thinking..."
        state["display_active"] = True
        state["scroll_offset"] = 0

    matched = None
    for keyword, action in COMMANDS.items():
        if keyword in text_input.lower():
            matched = action
            break

    if matched:
        response = matched.replace("_", " ")
        print(f'🤖 [LOCAL MOVEMENT COMMAND]: Executing wheel shift "{action}"')
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
        print(f'🧠 [GENERATED RESPONSE]: "{ai_response}"')
        with state_lock:
            state["response_text"] = ai_response
            state["speaking"] = True

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
            print("👤 [CAM FEED INTERACTION]: Target acquired. Triggering Greeting.")
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


# ══════════════════════════════════════════════════════
# FLASK WIRELESS COCKPIT SERVER (🕹️ FULLY INTEGRATED)
# ══════════════════════════════════════════════════════

HTML_COCKPIT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>AURA Wireless Cockpit</title>
    <style>
        body { margin: 0; background: #111; font-family: sans-serif; overflow: hidden; color: #fff; touch-action: none; }
        #video-container { width: 100vw; height: 55vh; background: #000; display: flex; justify-content: center; align-items: center; }
        #video-feed { width: 100%; height: 100%; object-fit: contain; border-bottom: 3px solid #0f0; }
        
        /* 🕹️ Clean Balanced D-Pad Grid Control Layout */
        #joystick-container { width: 100vw; height: 45vh; background: #1a1a1a; display: grid; 
                              grid-template-columns: repeat(3, 90px); grid-template-rows: repeat(3, 90px);
                              gap: 15px; justify-content: center; align-content: center; padding-bottom: 20px; }
        
        .control-btn { width: 90px; height: 90px; background: linear-gradient(145deg, #222, #333); 
                       border: 2px solid #0f0; border-radius: 50%; color: #0f0; font-size: 32px; font-weight: bold;
                       outline: none; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.5);
                       display: flex; justify-content: center; align-items: center;
                       -webkit-user-select: none; user-select: none; -webkit-tap-highlight-color: transparent; }
                       
        .control-btn:active { background: #0f0; color: #000; box-shadow: 0 0 20px #0f0; transform: scale(0.95); }
        .grid-center { grid-column: 2; grid-row: 2; border: 2px dashed #444; background: transparent; pointer-events: none; border-radius: 50%; }
        #btn-up { grid-column: 2; grid-row: 1; }
        #btn-left { grid-column: 1; grid-row: 2; }
        #btn-right { grid-column: 3; grid-row: 2; }
        #btn-down { grid-column: 2; grid-row: 3; }
    </style>
</head>
<body>
    <div id="video-container">
        <img id="video-feed" src="/video_feed">
    </div>
    <div id="joystick-container">
        <button id="btn-up" class="control-btn" ontouchstart="sendCmd('forward')" ontouchend="sendCmd('stop')">▲</button>
        <button id="btn-down" class="control-btn" ontouchstart="sendCmd('back')" ontouchend="sendCmd('stop')">▼</button>
        <button id="btn-left" class="control-btn" ontouchstart="sendCmd('left')" ontouchend="sendCmd('stop')">◀</button>
        <button id="btn-right" class="control-btn" ontouchstart="sendCmd('right')" ontouchend="sendCmd('stop')">▶</button>
    </div>
    <script>
        function sendCmd(action) {
            fetch('/control?action=' + action).catch(err => console.log(err));
        }
        // Block standard touch-scrolling to preserve smooth joystick responses
        document.addEventListener('touchmove', function(e) { e.preventDefault(); }, { passive: false });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_COCKPIT_TEMPLATE)


@app.route("/control")
def web_control():
    action = request.args.get("action")
    if action:
        if action in ["forward", "back", "left", "right", "stop"]:
            # Route movement pads strictly to local hardware actuators
            execute_command(COMMANDS[action])
        else:
            process_text_command(action, source_type="Web")
    return "OK"


def generate_web_frames():
    """Continuously yields the shared frame buffer as an HTTP multipart stream."""
    global global_web_frame
    while True:
        with frame_buffer_lock:
            if global_web_frame is None:
                time.sleep(0.01)
                continue
            # Fetch the active JPEG byte array copy
            frame_bytes = global_web_frame

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.04)  # ~25 FPS sync pacing loop


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_web_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


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
    except Exception:
        use_hardware_mic = False
        with state_lock:
            state["status"] = "AURA Sim Mic"
        try:
            fifo_file = open("/home/aura/AURA/mic_sim.pipe", "rb")
        except Exception:
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
    if len(text) > 65:
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
    if len(lines) > 4:
        scale = 0.32
        line_height = 12
    visible_lines = lines[scroll_offset:]
    y = start_y
    for i, line in enumerate(visible_lines):
        if i >= 2:
            break
        cv2.putText(
            frame, line, (start_x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA
        )
        y += line_height


def get_device_ip():
    """
    Dynamically fetches the current local network IP address of the Pi.
    Returns 'Offline' if not connected to a network router or phone hotspot.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Pinging a dummy public router address forces the OS to pick the active local network interface
        s.connect(("8.8.8.8", 80))
        ip_addr = s.getsockname()[0]
    except Exception:
        ip_addr = "Offline"
    finally:
        s.close()
    return ip_addr


def draw_overlay(frame, faces, heard, response, status, display_active, scroll_offset):
    h, w = frame.shape[:2]
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

    if display_active or heard or response:
        hud_h = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if heard:
            cv2.putText(
                frame, heard, (15, h - 62), font, 0.30, (0, 255, 255), 1, cv2.LINE_AA
            )
        if response:
            draw_scaled_multiline_text(
                frame=frame,
                text=response,
                start_x=15,
                start_y=h - 40,
                max_width=730,
                base_line_height=18,
                font=font,
                thickness=1,
                scroll_offset=scroll_offset,
            )

    font = cv2.FONT_HERSHEY_SIMPLEX
    device_ip = get_device_ip()

    ip_label = f"IP: {device_ip}"
    cv2.putText(
        frame, ip_label, (w - 170, 20), font, 0.30, (0, 0, 0), 1, cv2.LINE_AA
    )  # Cyan color for visibility

    # Push the status string down slightly or shift left to avoid overlap
    cv2.putText(
        frame, status, (w - 170, 38), font, 0.30, (200, 200, 200), 1, cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Trackers: {len(faces)}",
        (15, 20),
        font,
        0.30,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame


# ── Main Application Engine ───────────────────────────
def main():
    global global_web_frame
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

    # Spin up Flask Background Server instance
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=5000, threaded=True, use_reloader=False
        ),
        daemon=True,
    ).start()

    # Launch tracking loops
    threading.Thread(target=voice_thread, daemon=True).start()
    threading.Thread(target=optional_ssh_thread, daemon=True).start()
    threading.Thread(target=ultrasonic_watchdog_thread, daemon=True).start()

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

            # Render HUD bounding overlays onto the active frame
            frame = draw_overlay(
                frame,
                local_faces,
                heard,
                response,
                status,
                display_active,
                scroll_offset,
            )

            # Encode processed array down to compressed JPEG bytes and push to web buffer
            ret, encoded_buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if ret:
                with frame_buffer_lock:
                    global_web_frame = encoded_buffer.tobytes()

            # Resize to native 800x480 resolution and render to the HDMI monitor screen
            display_frame = cv2.resize(
                frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR
            )
            cv2.imshow("AURA", display_frame)

            # Safe interactive breakout trigger on local keyboard tap 'q'
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n👋 Manual application shutdown intercept triggered.")

    finally:
        # Failsafe execution termination lifecycle sequence
        gpio_cleanup()
        cv2.destroyAllWindows()
        picam2.stop()
        print("🛑 AURA core engine systems shut down safely.")


if __name__ == "__main__":
    main()
