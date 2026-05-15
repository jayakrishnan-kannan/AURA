# sim_mic.py
import wave
import time
import os

WAV_PATH = "/home/aura/AURA/test_input.wav"
PIPE_PATH = "/home/aura/AURA/mic_sim.pipe"

print("🎙️ Virtual Microphone Pipeline Simulator Initialized.")
print("👉 Waiting for main_aura.py to open the connection...")

while True:
    # Open the FIFO pipe. This blocks execution until main_aura.py starts reading it
    with open(PIPE_PATH, "wb") as fifo:
        print("🔗 Main application connected. Streaming simulated audio chunks...")

        while True:
            try:
                wf = wave.open(WAV_PATH, "rb")
            except FileNotFoundError:
                print(f"❌ Error: Please put a valid 16kHz WAV file at {WAV_PATH}")
                time.sleep(5)
                break

            # Verify formatting properties match Vosk constraints
            if wf.getframerate() != 16000 or wf.getnchannels() != 1:
                print(
                    "⚠️ Warning: For accurate results, your WAV file should be 16000Hz Mono."
                )

            while True:
                # Read 4000 frame audio bytes (Same block sizing used by PyAudio)
                data = wf.readframes(4000)
                if len(data) == 0:
                    print("🔄 WAV finished. Rewinding audio track loop...")
                    break

                try:
                    fifo.write(data)
                    fifo.flush()
                except BrokenPipeError:
                    print("🔌 Main application disconnected. Resetting channel...")
                    break

                # 💡 Crucial: Pause execution briefly to mimic the speed of real sound capture (16kHz).
                # 4000 frames at 16000Hz equals exactly 0.25 seconds of raw sound.
                time.sleep(0.25)

            wf.close()
            # If the pipe breaks (main app closed), break the internal loop to reset the pipe connection
            if not os.path.exists(PIPE_PATH):
                break
