from picamera2 import Picamera2
import cv2

# ── Constants ──────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 800, 480   # hardcoded — zero overhead
CASCADE_PATH = "/home/aura/AURA/model/haarcascade_frontalface_default.xml"

# ── Init ───────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()

# ── Fullscreen window ──────────────────────────────────
cv2.namedWindow("AURA", cv2.WINDOW_NORMAL)
cv2.moveWindow("AURA", 0, 0)
cv2.resizeWindow("AURA", DISPLAY_W, DISPLAY_H)
cv2.setWindowProperty("AURA", cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)

frame_count = 0
faces = []

# ── Main loop ──────────────────────────────────────────
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if frame_count % 5 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2,
            minNeighbors=4, minSize=(20, 20)
        )
        frame_count = 0
    frame_count += 1

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Upscale to display
    display_frame = cv2.resize(
        frame, (DISPLAY_W, DISPLAY_H),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imshow("AURA", display_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
