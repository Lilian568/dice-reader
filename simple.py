import argparse
import ast
import csv
import datetime
import os
import sys

import cv2
import numpy as np
from sklearn import cluster

LOG_FILE = "roll_log.csv"

# Path to the Tesseract executable — only needed for --mode numbers/auto.
# Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Number of consecutive identical frames before a roll is logged (~1.5 s at 100 ms/frame)
STABLE_FRAMES = 15

# --- Blob detector ---
# Tuned for standard dice dots: roughly circular, moderately sized, convex blobs.
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 80
params.maxArea = 4000

params.filterByCircularity = True
params.minCircularity = 0.55

params.filterByConvexity = True
params.minConvexity = 0.75

params.filterByInertia = True
params.minInertiaRatio = 0.4

detector = cv2.SimpleBlobDetector_create(params)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def get_blobs(frame):
    blurred = cv2.medianBlur(frame, 7)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # First try dark blobs (standard dice: dark dots on white/light face)
    blobs = detector.detect(gray)
    if not blobs:
        # Fallback: invert for colored dice with white dots (e.g. red die)
        blobs = detector.detect(cv2.bitwise_not(gray))

    # Remove isolated blobs (shadows, reflections) that have no neighbor within 150 px.
    # Skip this when only 1 blob found — a die showing "1" has no neighbors by definition.
    if len(blobs) > 1:
        pts = np.asarray([b.pt for b in blobs])
        kept = [b for i, b in enumerate(blobs)
                if np.sum(np.linalg.norm(pts - pts[i], axis=1) < 150) >= 2]
        blobs = kept if kept else blobs

    return blobs


_FACE_NAMES = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_BOX_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 165, 255),  # orange
    (128, 0, 128),  # purple
    (0, 255, 255),  # yellow
]


def get_dice_from_blobs(blobs):
    valid = [(b.pt, b.size) for b in blobs if b.pt is not None]
    if not valid:
        return []

    pts = np.asarray([v[0] for v in valid])
    sizes = np.asarray([v[1] for v in valid])

    def _build_dice(labels):
        dice = []
        for i in range(int(max(labels)) + 1):
            mask = labels == i
            group = pts[mask]
            count = len(group)
            if not (1 <= count <= 6):
                continue
            centroid = np.mean(group, axis=0)
            pad = int(np.mean(sizes[mask]) * 1.5) + 20
            x1 = int(np.min(group[:, 0])) - pad
            y1 = int(np.min(group[:, 1])) - pad
            x2 = int(np.max(group[:, 0])) + pad
            y2 = int(np.max(group[:, 1])) + pad
            dice.append([count, *centroid, x1, y1, x2, y2])
        return dice

    # Try progressively tighter clustering.
    # A large eps merges dots from multiple visible die faces (3D/angled photos);
    # shrinking eps separates them so each face forms its own valid cluster (1-6).
    for eps in [70, 55, 40]:
        dice = _build_dice(cluster.DBSCAN(eps=eps, min_samples=1).fit(pts).labels_)
        if dice:
            break

    # Noise filter: single-dot clusters that are far from any larger cluster are
    # almost always side-face artifacts, not real "one" dice.
    if any(d[0] >= 2 for d in dice):
        large = [d for d in dice if d[0] >= 2]
        filtered = large[:]
        for d in dice:
            if d[0] == 1:
                d_pos = np.array([d[1], d[2]])
                min_dist = min(
                    np.linalg.norm(d_pos - np.array([ld[1], ld[2]])) for ld in large
                )
                if min_dist < 250:
                    filtered.append(d)
        dice = filtered

    return dice


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay_info(frame, dice, blobs, history):
    h, w = frame.shape[:2]

    # Draw a circle around each detected dot
    for b in blobs:
        pos = b.pt
        cv2.circle(frame, (int(pos[0]), int(pos[1])), int(b.size / 2), (0, 220, 255), 2)

    # Draw bounding box and label for each die
    for idx, d in enumerate(dice):
        count = d[0]
        x1 = max(0, d[3])
        y1 = max(0, d[4])
        x2 = min(w - 1, d[5])
        y2 = min(h - 1, d[6])
        color = _BOX_COLORS[idx % len(_BOX_COLORS)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = _FACE_NAMES.get(count, str(count))
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        tag_y = max(y1 - 4, lh + 4)
        cv2.rectangle(frame, (x1, tag_y - lh - baseline - 4), (x1 + lw + 4, tag_y), color, -1)
        cv2.putText(frame, label, (x1 + 2, tag_y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Total score banner
    if dice:
        total = sum(d[0] for d in dice)
        banner = f"Total score = {total} points"
        cv2.putText(frame, banner, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        # Show each die's count below the banner
        for idx, d in enumerate(dice):
            color = _BOX_COLORS[idx % len(_BOX_COLORS)]
            cv2.putText(frame, str(d[0]), (10 + idx * 30, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Roll history panel — last 5 entries
    if history:
        cv2.putText(frame, "History:", (10, 100),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (200, 200, 200), 1)
        for i, entry in enumerate(reversed(history[-5:])):
            cv2.putText(frame, entry, (10, 100 + 26 * (i + 1)),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_roll(dice, history):
    values = [d[0] for d in dice]
    total = sum(values)
    now = datetime.datetime.now()
    ts = now.strftime("%H:%M:%S")

    history.append(f"{ts}  {values}  = {total}")

    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([now.isoformat(), total, str(values)])

    print(f"[{ts}] Roll logged: {values} = {total}")


# ---------------------------------------------------------------------------
# Source abstraction
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _is_image(path):
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _is_video(path):
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def open_source(source):
    """
    Returns (cap, is_picam2, is_image, image_frame).
    Caller is responsible for releasing cap when is_picam2=False and is_image=False.
    """
    if source == "picamera":
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            picam2.configure(picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": (640, 480)}))
            picam2.start()
            return picam2, True, False, None
        except Exception as e:
            print(f"Picamera2 not available: {e}")
            sys.exit(1)

    if os.path.isfile(source) and _is_image(source):
        frame = cv2.imread(source)
        if frame is None:
            print(f"Could not read image: {source}")
            sys.exit(1)
        return None, False, True, frame

    # Webcam index, video file, or explicit "webcam" keyword
    if source == "webcam":
        index = 0
    elif source.isdigit():
        index = int(source)
    elif os.path.isfile(source) and _is_video(source):
        index = source
    else:
        print(f"Unknown source '{source}'. Use 'webcam', 'picamera', an image path, or a video path.")
        sys.exit(1)

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap, False, False, None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def show_stats():
    if not os.path.exists(LOG_FILE):
        print("No log file found — roll some dice first.")
        return

    totals = []
    all_values = []

    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                totals.append(int(row["total"]))
                vals = ast.literal_eval(row["values"])
                all_values.extend(vals)
            except (ValueError, KeyError, SyntaxError):
                continue

    if not totals:
        print("No rolls recorded yet.")
        return

    print("\n=== Dice Roll Statistics ===")
    print(f"Total rolls logged : {len(totals)}")
    print(f"Average total      : {sum(totals) / len(totals):.2f}")
    print(f"Highest total      : {max(totals)}")
    print(f"Lowest total       : {min(totals)}")

    if all_values:
        print(f"\nIndividual die face stats ({len(all_values)} dice across all rolls):")
        print(f"  Average face value : {sum(all_values) / len(all_values):.2f}")
        print(f"  Min / Max face     : {min(all_values)} / {max(all_values)}")

        dist = {}
        for v in all_values:
            dist[v] = dist.get(v, 0) + 1
        max_count = max(dist.values())
        bar_width = 30

        print("\n  Face distribution:")
        for k in sorted(dist):
            filled = round(dist[k] / max_count * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"    {k}: [{bar}] {dist[k]}")


# ---------------------------------------------------------------------------
# OCR-based number dice detection
# ---------------------------------------------------------------------------

def detect_number_dice(frame):
    """
    Detect dice that show printed digits (1-6) instead of pips.
    Requires Tesseract OCR to be installed on the system.
    """
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    except ImportError:
        print("pytesseract not installed. Run: pip install pytesseract")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    dice = []
    # Try both polarities: dark digits on light background, and light digits on dark
    for src in [gray, cv2.bitwise_not(gray)]:
        blurred = cv2.GaussianBlur(src, (3, 3), 0)
        try:
            data = pytesseract.image_to_data(
                blurred,
                config="--psm 11 --oem 3 -c tessedit_char_whitelist=123456",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue

        for i, text in enumerate(data["text"]):
            text = text.strip()
            if text not in ("1", "2", "3", "4", "5", "6"):
                continue
            if int(data["conf"][i]) < 40:
                continue
            x = data["left"][i]
            y = data["top"][i]
            bw = data["width"][i]
            bh = data["height"][i]
            if bw < 10 or bh < 10:
                continue
            # Reject detections that cover the whole image (false positives)
            if bw * bh > w * h * 0.8:
                continue
            pad = 20
            cx = x + bw // 2
            cy = y + bh // 2
            dice.append([int(text), cx, cy,
                         max(0, x - pad), max(0, y - pad),
                         min(w - 1, x + bw + pad), min(h - 1, y + bh + pad)])

        if dice:
            return dice

    return dice


# ---------------------------------------------------------------------------
# Main video loop
# ---------------------------------------------------------------------------

def run_video(source, mode="pips"):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "total", "values"])

    cap, is_picam2, is_image, image_frame = open_source(source)

    history = []
    last_values = []
    stable_count = 0

    cv2.namedWindow("Dice Reader", cv2.WINDOW_NORMAL)

    def read_frame():
        if is_picam2:
            raw = cap.capture_array()
            # Picamera2 XRGB8888 → 4 channels; convert to BGR
            if raw.ndim == 3 and raw.shape[2] == 4:
                return True, cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            return True, raw
        if is_image:
            return True, image_frame.copy()
        return cap.read()

    while True:
        ok, frame = read_frame()
        if not ok or frame is None:
            break

        if mode == "numbers":
            blobs = []
            dice = detect_number_dice(frame)
        elif mode == "auto":
            blobs = get_blobs(frame)
            dice = get_dice_from_blobs(blobs)
            if not dice:
                dice = detect_number_dice(frame)
        else:  # pips
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
        cv2.imshow("Dice Reader", frame)

        # For static images: wait indefinitely for any key press
        delay = 0 if is_image else 100
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q") or key == 27:  # q or Esc
            break

    if is_picam2:
        cap.stop()
    elif not is_image:
        cap.release()

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dice Reader — detects and logs dice rolls from a camera or file.")
    parser.add_argument(
        "--source",
        default="webcam",
        metavar="SOURCE",
        help=(
            "Input source (default: webcam). Options:\n"
            "  webcam        — default system camera (index 0)\n"
            "  0, 1, 2, …   — specific camera index\n"
            "  picamera      — Raspberry Pi camera via Picamera2\n"
            "  path/to/img   — static image file (.jpg, .png, …)\n"
            "  path/to/vid   — video file (.mp4, .avi, …)"
        ),
    )
    parser.add_argument(
        "--mode",
        default="pips",
        choices=["pips", "numbers", "auto"],
        help=(
            "Detection mode (default: pips).\n"
            "  pips    — blob detection for traditional dot dice\n"
            "  numbers — OCR for dice with printed digits (requires Tesseract)\n"
            "  auto    — try pips first, fall back to OCR"
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics from roll_log.csv and exit.",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    run_video(args.source, mode=args.mode)


if __name__ == "__main__":
    main()
