import cv2

DISPLAY_W = 800
DISPLAY_H = 480

video = cv2.VideoCapture("/home/aura/AURA/boot_eyes.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

delay = int(1000 / fps)

cv2.namedWindow("AURA", cv2.WINDOW_NORMAL)

cv2.resizeWindow("AURA", DISPLAY_W, DISPLAY_H)
cv2.moveWindow("AURA", 0, 0)

cv2.setWindowProperty("AURA", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

    cv2.imshow("AURA", frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
