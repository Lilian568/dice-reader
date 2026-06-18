import csv
import datetime
import os

import cv2
import numpy as np
from picamera2 import Picamera2
from sklearn import cluster

LOG_FILE = "roll_log.csv"
# Number of consecutive identical frames before a roll is logged (~1.5 s at 100 ms/frame)
STABLE_FRAMES = 15

params = cv2.SimpleBlobDetector_Params()
params.filterByInertia
params.minInertiaRatio = 0.6
detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    return detector.detect(frame_gray)


def get_dice_from_blobs(blobs):
    X = [b.pt for b in blobs if b.pt is not None]
    X = np.asarray(X)

    if len(X) == 0:
        return []

    # Note: eps=70 groups dots belonging to the same die; adjust if dice are very close/far
    clustering = cluster.DBSCAN(eps=70, min_samples=1).fit(X)
    num_dice = max(clustering.labels_) + 1

    dice = []
    for i in range(num_dice):
        X_dice = X[clustering.labels_ == i]
        centroid = np.mean(X_dice, axis=0)
        dice.append([len(X_dice), *centroid])

    return dice


def overlay_info(frame, dice, blobs, history):
    # Draw a circle around each detected dot
    for b in blobs:
        pos = b.pt
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(b.size / 2), (255, 0, 0), 2)

    # Draw each die's value at its centroid
    for d in dice:
        textsize = cv2.getTextSize(str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2), int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    # Total sum of all dice
    if dice:
        total = sum(d[0] for d in dice)
        cv2.putText(frame, f"Total: {total}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)

    # Roll history panel — last 5 entries
    if history:
        cv2.putText(frame, "History:", (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 200, 200), 1)
        for i, entry in enumerate(reversed(history[-5:])):
            cv2.putText(frame, entry, (10, 90 + 28 * (i + 1)),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1)


def log_roll(dice, history):
    values = [d[0] for d in dice]
    total = sum(values)
    now = datetime.datetime.now()
    ts = now.strftime("%H:%M:%S")

    history.append(f"{ts}  {values}  = {total}")

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([now.isoformat(), total, str(values)])

    print(f"[{ts}] Roll logged: {values} = {total}")


def main():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "total", "values"])

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}))
    picam2.start()

    history = []
    last_values = []
    stable_count = 0

    cv2.namedWindow("Simple Dice Reader", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Simple Dice Reader", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        frame = picam2.capture_array()
        blobs = get_blobs(frame)
        dice = get_dice_from_blobs(blobs)

        values = [d[0] for d in dice]

        if values and values == last_values:
            stable_count += 1
            if stable_count == STABLE_FRAMES:
                log_roll(dice, history)
        else:
            stable_count = 0

        last_values = values

        overlay_info(frame, dice, blobs, history)
        cv2.imshow("Simple Dice Reader", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
