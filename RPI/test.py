import cv2
import time
from picamera2 import Picamera2

# Initialize video capture
picam2 = Picamera2()
height = 480
width = 640
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
picam2.start()

#capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
#capture.set(cv2.CAP_PROP_FPS, 30)  # Requesting 30 FPS from the camera
 
# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
 
# Current mode
mode = "normal"
 
# FPS calculation
prev_time = time.time()
 
# Video writer initialization with explicit FPS
save_fps = 15.0  # Adjust this based on your actual processing speed
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, save_fps, (1024, 576))
 
while True:
    #ret, frame = capture.read()
    #if not ret:
        #break
 
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    #display_frame = picam2.capture_array()
 
    if mode == "threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, display_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
 
    elif mode == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
 
    elif mode == "bg_sub":
        fg_mask = bg_subtractor.apply(frame)
        display_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
 
    elif mode == "contour":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        display_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
 
    # Calculate actual processing FPS
    curr_time = time.time()
    processing_fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
 
    # Display actual processing FPS
    cv2.putText(
        display_frame, f"FPS: {int(processing_fps)} Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
 
    # Write frame to video
    out.write(display_frame)
 
    # Show video
    cv2.imshow("Live Video", display_frame)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("t"):
        mode = "threshold"
    elif key == ord("e"):
        mode = "edge"
    elif key == ord("b"):
        mode = "bg_sub"
    elif key == ord("c"):
        mode = "contour"
    elif key == ord("n"):
        mode = "normal"
    elif key == ord("q"):
        break
 
# Clean up
capture.release()
out.release()
cv2.destroyAllWindows()
