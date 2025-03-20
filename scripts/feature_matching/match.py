import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Match features between two images based on intensities.")
    parser.add_argument("image1", type=str, help="Path to the first image", nargs='?', default='data/firefly_assets/left_img_rectified.png')
    parser.add_argument("image2", type=str, help="Path to the second image", nargs='?', default='data/ximea_assets/img.png')
    args = parser.parse_args()

    # Load the images
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # Convert images to grayscale (using intensities)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    scale_factor = np.mean([img2.shape[0] / img1.shape[0], img2.shape[1] / img1.shape[1]])
    gray1 = cv2.resize(img1, None, fx=scale_factor, fy=scale_factor)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the contrast
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray1 = clahe.apply(gray1)
    # gray2 = clahe.apply(gray2)

    # gray1 = cv2.equalizeHist(gray1)
    # gray2 = cv2.equalizeHist(gray2)

    # ### Initialize ORB detector (suitable for intensity-based feature detection) ###
    # orb = cv2.ORB_create()

    # # Detect keypoints and compute descriptors for both images
    # kp1, des1 = orb.detectAndCompute(gray1, None)
    # kp2, des2 = orb.detectAndCompute(gray2, None)

    # # Create a Brute Force matcher with Hamming distance (appropriate for ORB)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)

    # # Sort the matches based on distance (i.e., quality of the match)
    # matches = sorted(matches, key=lambda x: x.distance)

    # # Optionally, keep only the best matches (for example, top 50)
    # num_matches_to_draw = 10
    # good_matches = matches[:num_matches_to_draw]

    # Compute edge maps using Canny to emphasize structural features
    edges1 = cv2.Canny(gray1, 100, 300, apertureSize=3, L2gradient=True) # L2 makes a big improvement
    edges2 = cv2.Canny(gray2, 30, 100, apertureSize=3, L2gradient=True)

    # Optionally, display edge maps to verify
    cv2.imshow("Edges1", edges1)
    cv2.imshow("Edges2", edges2)
    cv2.waitKey(0)

    # Use SIFT for robust feature detection on the edge maps (you can also combine with the original gray images)
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.06, edgeThreshold=20, sigma=1.6)
    kp1_scaled, des1 = sift.detectAndCompute(edges1, None)
    kp2, des2 = sift.detectAndCompute(edges2, None)

    # Use BFMatcher with L2 norm (suitable for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe’s ratio test to filter out weak matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    matched_img = cv2.drawMatches(gray1, kp1_scaled, gray2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("Feature Matches (Scaled)", matched_img)
    cv2.waitKey(0)

    # Convert keypoints from the scaled image back to original image coordinates
    kp1_original = []
    for kp in kp1_scaled:

        # Create a new keypoint with the scaled coordinates reversed
        kp_orig = cv2.KeyPoint(kp.pt[0] / scale_factor,
                       kp.pt[1] / scale_factor,
                       kp.size,
                       kp.angle,
                       kp.response,
                       kp.octave,
                       kp.class_id)
        
        kp1_original.append(kp_orig)

    # Draw matches using the original keypoints for image1 and the detected keypoints for image2
    matched_img = cv2.drawMatches(img1, kp1_original, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the final matched image
    cv2.imshow("Feature Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
