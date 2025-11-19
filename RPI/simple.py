import cv2
import numpy as np
from picamera2 import Picamera2
from sklearn import cluster

params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.6

detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs


def get_dice_from_blobs(blobs):
    # Get centroids of all blobs
    X = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        clustering = cluster.DBSCAN(eps=70, min_samples=1).fit(X)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []

        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])

        return dice

    else:
        return []


def overlay_info(frame, dice, blobs):
    # Overlay blobs
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice number
    for d in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


def main():
    # Initialize a video feed
    #cap = cv2.VideoCapture(0)
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 960)}))
    picam2.start()

    last_blobs_count = -1
    last_dice = []

    while(True):
        print_update = False

        #ret, frame = cap.read()
        frame = picam2.capture_array()

        blobs = get_blobs(frame)
        if len(blobs) != last_blobs_count:
            print_update = True
        last_blobs_count = len(blobs)

        dice = get_dice_from_blobs(blobs)
        if len(dice) != len(last_dice):
            print_update = True
        else:
            for i in range(len(dice)):
                if dice[i][0] != last_dice[i][0]:
                    print_update = True
        last_dice = dice

        if print_update:
            print("Total: ", len(blobs))
            print("Count: ", len(dice))
            for d in dice:
                print("Dice: ", d[0])

        out_frame = overlay_info(frame, dice, blobs)

        cv2.imshow("Dice Reader", frame)

        res = cv2.waitKey(100)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
