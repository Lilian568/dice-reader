#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "dbscan.hpp"

cv::Ptr<cv::SimpleBlobDetector> create_blob_detector() {
    cv::SimpleBlobDetector::Params params;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.6f;

    return cv::SimpleBlobDetector::create(params);
}

std::vector<cv::KeyPoint> gather_blobs(cv::Ptr<cv::SimpleBlobDetector> detector, cv::Mat &frame) {
    cv::Mat blur_frame, gray_frame;
    cv::medianBlur(frame, blur_frame, 7);
    cv::cvtColor(blur_frame, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray_frame, keypoints);
    return keypoints;
}

struct DBSCAN {
    double eps;
    int minPts;

    static double dist(const cv::Point2f &a, const cv::Point2f &b) { return cv::norm(a - b); }

    std::vector<int> fit(const std::vector<cv::Point2f> &points) {
        const int n = points.size();
        std::vector<int> labels(n, -1);
        int clusterId = 0;

        for (int i = 0; i < n; i++) {
            if (labels[i] != -1) continue;

            // Find neighbors
            std::vector<int> neighbors;
            for (int j = 0; j < n; j++)
                if (dist(points[i], points[j]) <= eps) neighbors.push_back(j);

            if (neighbors.size() < minPts) {
                labels[i] = -1; // Noise
                continue;
            }

            // Assign cluster
            labels[i] = clusterId;

            for (size_t k = 0; k < neighbors.size(); k++) {
                int idx = neighbors[k];

                if (labels[idx] == -1) labels[idx] = clusterId;

                if (labels[idx] != -1) continue;

                labels[idx] = clusterId;

                // Check neighbor's neighbors
                std::vector<int> neighbors2;
                for (int j = 0; j < n; j++)
                    if (dist(points[idx], points[j]) <= eps) neighbors2.push_back(j);

                if (neighbors2.size() >= minPts)
                    neighbors.insert(neighbors.end(), neighbors2.begin(), neighbors2.end());
            }

            clusterId++;
        }

        return labels;
    }
};

// Get dice from blobs
struct DiceInfo {
    int dots;
    float cx;
    float cy;
};

std::vector<DiceInfo> get_dice_from_blobs(const std::vector<cv::KeyPoint> &blobs, const double eps) {

    std::vector<cv::Point2f> pts;
    for (const auto &b : blobs)
        pts.push_back(b.pt);

    if (pts.empty()) return {};

    DBSCAN db(eps, 1);
    std::vector<int> labels = db.fit(pts);

    int num_clusters = *max_element(labels.begin(), labels.end()) + 1;

    std::vector<DiceInfo> dice;

    for (int c = 0; c < num_clusters; c++) {
        std::vector<cv::Point2f> group;

        for (size_t i = 0; i < pts.size(); i++)
            if (labels[i] == c) group.push_back(pts[i]);

        // Compute centroid
        cv::Point2f sum(0, 0);
        for (auto &p : group)
            sum += p;

        cv::Point2f centroid = sum * (1.0 / group.size());

        dice.push_back({(int)group.size(), centroid.x, centroid.y});
    }

    return dice;
}

void overlay_info(cv::Mat &frame, const std::vector<DiceInfo> &dice, const std::vector<cv::KeyPoint> &blobs) {
    // Draw blob circles
    for (const auto &b : blobs) {
        cv::Point center((int)b.pt.x, (int)b.pt.y);
        int radius = (int)(b.size * 0.5);

        cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
    }

    // Draw dice numbers
    for (const auto &d : dice) {
        std::string text = std::to_string(d.dots);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN, 3, 2, &baseline);

        cv::Point pos((int)(d.cx - text_size.width / 2), (int)(d.cy + text_size.height / 2));

        cv::putText(frame, text, pos, cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <epsilon>" << std::endl;
        return 1;
    }

    double eps = std::stod(argv[1]);

    // Initialize the camera
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Check if the camera was opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error opening the camera" << std::endl;
        return 1;
    }

    // Set up SimpleBlobDetector
    cv::Ptr<cv::SimpleBlobDetector> detector = create_blob_detector();

    // Main loop
    cv::namedWindow("Dice Reader", cv::WINDOW_NORMAL);
    cv::Mat output;
    std::vector<cv::KeyPoint> keypoints = {};
    std::vector<DiceInfo> dice = {};
    int counter = 0;

    while (cap.read(output)) {
        if (counter == 5) {
            std::vector<cv::KeyPoint> keypoints = gather_blobs(detector, output);
            std::cout << keypoints.size() << std::endl;

            dice = get_dice_from_blobs(keypoints, eps);

            counter = 0;
        } else {
            counter++;
        }

        overlay_info(output, dice, keypoints);

        cv::imshow("Dice Reader", output);

        if (cv::waitKey(1) == 'q') { break; }
    }

    // Clean up
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
