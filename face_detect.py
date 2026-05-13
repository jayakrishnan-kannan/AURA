from picamera2 import Picamera2
import cv2
import time

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "/home/pi/AURA/model/haarcascade_frontalface_default.xml"
)

picam2 = Picamera2()

# LOW resolution + direct BGR output
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"}
)

picam2.configure(config)
picam2.start()

last_detect_time = 0
faces = []

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
   

    # Run detection only every 0.3 sec
    current_time = time.time()

    if current_time - last_detect_time > 0.3:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        last_detect_time = current_time

    # Draw previous detections
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Small sleep reduces CPU a LOT
    time.sleep(0.02)

cv2.destroyAllWindows()
picam2.stop()
