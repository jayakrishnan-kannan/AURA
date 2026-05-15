# test_voice.py — runs without microphone
# Uses a pre-recorded WAV file to test vosk STT pipeline

from vosk import Model, KaldiRecognizer
import wave
import json
import os

MODEL_PATH = "/home/aura/AURA/model/vosk-model-small-en-us-0.15"

# ── Create a test wav using espeak ─────────────────────
# This generates "move forward" as a wav file to feed into vosk
os.system('espeak-ng "move forward" -w /home/aura/AURA/test_input.wav')

# ── Load model ─────────────────────────────────────────
model = Model(MODEL_PATH)

# ── Run STT on wav file ────────────────────────────────
wf = wave.open("/home/aura/AURA/test_input.wav", "rb")
rec = KaldiRecognizer(model, wf.getframerate())

result_text = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)

result = json.loads(rec.FinalResult())
result_text = result.get("text", "")
print(f"STT heard: '{result_text}'")

# ── Intent matching ────────────────────────────────────
COMMANDS = {
    "forward": "move_forward",
    "backward": "move_backward",
    "left": "turn_left",
    "right": "turn_right",
    "stop": "stop_motors",
}

matched = None
for keyword, action in COMMANDS.items():
    if keyword in result_text:
        matched = action
        break

if matched:
    print(f"Command matched: {matched}")
else:
    print("No command matched")
