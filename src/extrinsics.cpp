#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/aruco/charuco.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Holds camera parameters loaded from YAML.
 */
struct CameraParams {
    cv::Mat cameraMatrix;       // 3x3
    cv::Mat distCoeffs;         // Nx1
    cv::Mat rectification;      // 3x3
    cv::Mat projection;         // 3x4
    int width = 0;
    int height = 0;
};

/**
 * @brief Load camera parameters from a YAML file.
 *
 * The YAML is expected to have:
 *   camera_matrix:
 *     data: [...]
 *   distortion_coefficients:
 *     data: [...]
 *   rectification_matrix:
 *     data: [...]
 *   projection_matrix:
 *     data: [...]
 *   image_width: ...
 *   image_height: ...
 */
CameraParams loadCameraParams(const std::string& yamlFile)
{
    YAML::Node config = YAML::LoadFile(yamlFile);
    CameraParams cp;

    // camera_matrix -> data -> vector<float>
    {
        auto cmData = config["camera_matrix"]["data"].as<std::vector<float>>();
        cv::Mat cm = cv::Mat(cmData).reshape(1, 3).clone(); // Flatten into 3x3
        cp.cameraMatrix = cm;
    }
    // distortion_coefficients -> data -> vector<float>
    {
        auto distData = config["distortion_coefficients"]["data"].as<std::vector<float>>();
        cv::Mat dc = cv::Mat(distData).reshape(1, 1).clone(); // 1 row, N columns
        cp.distCoeffs = dc;
    }
    // rectification_matrix -> data -> vector<float>
    {
        auto rectData = config["rectification_matrix"]["data"].as<std::vector<float>>();
        cv::Mat rm = cv::Mat(rectData).reshape(1, 3).clone();
        cp.rectification = rm;
    }
    // projection_matrix -> data -> vector<float>
    {
        auto projData = config["projection_matrix"]["data"].as<std::vector<float>>();
        cv::Mat pm = cv::Mat(projData).reshape(1, 3).clone(); // shape (3,4)
        cp.projection = pm;
    }
    cp.width  = config["image_width"].as<int>();
    cp.height = config["image_height"].as<int>();

    return cp;
}

/**
 * @brief Compute the relative transform (rotation, translation) from camera0 -> camera1,
 *        given each camera's pose w.r.t. the same board.
 *
 *        R_rel = R2 * R1^T
 *        t_rel = t2 - R_rel * t1
 */
void computeRelativeTransform(const cv::Mat &rvec1,
                              const cv::Mat &tvec1,
                              const cv::Mat &rvec2,
                              const cv::Mat &tvec2,
                              cv::Mat &R_rel,
                              cv::Mat &t_rel)
{
    cv::Mat R1, R2;
    cv::Rodrigues(rvec1, R1); // 3x3
    cv::Rodrigues(rvec2, R2); // 3x3

    R_rel = R2 * R1.t();      // 3x3
    t_rel = tvec2 - R_rel * tvec1;
}

/**
 * @brief Detect a ChArUco board in an image, then estimate its pose (classic C++ API).
 *
 * Steps:
 *   1) detectMarkers
 *   2) refineDetectedMarkers (optional)
 *   3) interpolateCornersCharuco
 *   4) estimatePoseCharucoBoard
 *
 * @param image BGR or grayscale
 * @param board A CharucoBoard describing squaresX, squaresY, markerLength, etc.
 * @param cameraMatrix 3x3 intrinsics
 * @param distCoeffs Nx1
 * @param rvec [out] rotation
 * @param tvec [out] translation
 * @return true if pose is found, else false
 */
bool detectCharucoPose(const cv::Mat &image,
                       const cv::Ptr<cv::aruco::CharucoBoard> &board,
                       const cv::Mat &cameraMatrix,
                       const cv::Mat &distCoeffs,
                       cv::Mat &rvec,
                       cv::Mat &tvec)
{
    // 1) Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // 2) Detect the ArUco markers
    std::vector<std::vector<cv::Point2f>> markerCorners, rejected;
    std::vector<int> markerIds;

    cv::Ptr<cv::aruco::Dictionary> dictionary = board->dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

    cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds, detectorParams, rejected);
    if (markerIds.empty()) {
        std::cerr << "[WARN] No ArUco markers detected.\n";
        return false;
    }

    // (Optional) refine the detection
    cv::aruco::refineDetectedMarkers(gray, board, markerCorners, markerIds, rejected, cameraMatrix, distCoeffs);

    // 3) Interpolate ChArUco corners
    cv::Mat charucoCorners, charucoIds;
    cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, gray, board,
                                         charucoCorners, charucoIds);

    if (charucoIds.total() < 4) {
        std::cerr << "[WARN] Not enough ChArUco corners to estimate pose.\n";
        return false;
    }

    // 4) Estimate pose
    bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners,
                                                     charucoIds,
                                                     board,
                                                     cameraMatrix,
                                                     distCoeffs,
                                                     rvec,
                                                     tvec);
    if (!valid) {
        std::cerr << "[WARN] estimatePoseCharucoBoard() failed.\n";
        return false;
    }

    // OPTIONAL: draw markers / corners for visualization
    //   drawDetectedMarkers(), drawDetectedCornersCharuco(), etc.
    cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
    // cv::aruco::drawDetectedCornersCharuco(image, charucoCorners, charucoIds, cv::Scalar(0, 255, 0));
    cv::imshow("Detected ChArUco", image);
    cv::waitKey(0);
    cv::destroyWindow("Detected ChArUco");

    return true;
}

int main(int argc, char** argv)
{
    // For demonstration, we won't parse advanced ROS arguments.
    // We'll treat this as a standalone node in the image_fusion package.

    std::string img0_path   = "image_data/firefly_left/1741990075_217062601_rect_firefly_left.png";
    std::string img1_path   = "image_data/ximea/675/1741990075_217062601_rect_ximea_675.png";
    std::string calib0_path = "src/image_fusion/calibration_data/firefly_left.yaml";
    std::string calib1_path = "src/image_fusion/calibration_data/ximea.yaml";

    // Load images
    cv::Mat img0 = cv::imread(img0_path, cv::IMREAD_COLOR);
    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_COLOR);
    if (img0.empty() || img1.empty()) {
        std::cerr << "[ERROR] Could not load one of the images. Check file paths.\n";
        return 1;
    }

    // Load camera params
    CameraParams cam0 = loadCameraParams(calib0_path);
    CameraParams cam1 = loadCameraParams(calib1_path);

    // Create a ChArUco board
    // squaresX=6, squaresY=4, squareLength=0.04, markerLength=0.03
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
    cv::Ptr<cv::aruco::CharucoBoard> board =
        cv::aruco::CharucoBoard::create(6, 4, 0.04f, 0.03f, dictionary);

    // Detect board pose in each image
    cv::Mat rvec0, tvec0;
    bool success0 = detectCharucoPose(img0, board, cam0.cameraMatrix, cam0.distCoeffs, rvec0, tvec0);

    cv::Mat rvec1, tvec1;
    bool success1 = detectCharucoPose(img1, board, cam1.cameraMatrix, cam1.distCoeffs, rvec1, tvec1);

    if (!success0 || !success1) {
        std::cerr << "[ERROR] Could not detect pose in one or both images.\n";
        return 1;
    }

    // Compute the relative transformation (cam0 -> cam1)
    cv::Mat R_rel, t_rel;
    computeRelativeTransform(rvec0, tvec0, rvec1, tvec1, R_rel, t_rel);

    std::cout << "Relative Rotation Matrix (Camera 0 -> Camera 1):\n" << R_rel << std::endl;
    std::cout << "\nRelative Translation Vector (Camera 0 -> Camera 1):\n" << t_rel << std::endl;

    // Draw coordinate axes for visualization
    float axisLen = 0.1f; // 10 cm
    cv::drawFrameAxes(img0, cam0.cameraMatrix, cam0.distCoeffs, rvec0, tvec0, axisLen, 2);
    cv::drawFrameAxes(img1, cam1.cameraMatrix, cam1.distCoeffs, rvec1, tvec1, axisLen, 2);

    cv::imshow("Camera 0 Pose", img0);
    cv::imshow("Camera 1 Pose", img1);
    cv::waitKey(0);

    return 0;
}
