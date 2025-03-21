#!/usr/bin/env python3

import cv2
import numpy as np
from fpdf import FPDF

def main():
    # ---------------------------------------------------
    # 1) DEFINE CHARUCO PARAMETERS
    # ---------------------------------------------------
    # Dictionary: Must match the board you really want
    ARUCO_DICT = cv2.aruco.DICT_5X5_50
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # Charuco dimensions: squaresX vs. squaresY
    # Make sure these match the number of chessboard squares
    # horizontally and vertically you intend to detect.
    squaresX = 6
    squaresY = 4

    # Physical sizes in meters (or any consistent unit)
    # The ratio markerLength / squareLength must match your real board
    squareLength = 0.04   # e.g. 40 mm
    markerLength = 0.03   # e.g. 30 mm

    # ---------------------------------------------------
    # 2) CREATE AND SAVE SYNTHETIC BOARD IMAGE
    # ---------------------------------------------------
    board = cv2.aruco.CharucoBoard(
        (squaresX, 
        squaresY), 
        squareLength, 
        markerLength, 
        dictionary
    )
    # Draw the board at a chosen resolution (pixels x pixels)
    board_size = 100
    board_img = board.generateImage((board_size*squaresX, board_size*squaresY))

    # Save for inspection
    out_filename = "charuco_test.png"
    cv2.imwrite(out_filename, board_img)
    print(f"[INFO] Saved synthetic board image as {out_filename}")

    # Create an A4 PDF and place the image on it
    page_height_mm = 210
    page_width_mm = 297
    pdf = FPDF(orientation='landscape', unit='mm', format=(page_height_mm, page_width_mm))
    pdf.add_page()

    # Center the board on the 8.5x11 inch page
    board_width_mm = (squareLength * 1000) * squaresX  # Convert to mm
    board_height_mm = (squareLength * 1000) * squaresY  # Convert to mm
    x_offset = (page_width_mm - board_width_mm) / 2
    y_offset = (page_height_mm - board_height_mm) / 2
    pdf.image(out_filename, x=x_offset, y=y_offset, w=board_width_mm, h=board_height_mm)
    pdf.output(f"charuco_{page_width_mm}_{page_height_mm}_{int(squareLength*1000)}_{int(markerLength*1000)}_5X5.pdf")
    print("[INFO] Generated 8x11 inch PDF as charuco_test.pdf")

    # Optionally show the board
    cv2.imshow("Synthetic Charuco Board", board_img)
    cv2.waitKey(1000)  # Display for 1 second
    cv2.destroyAllWindows()

    # ---------------------------------------------------
    # 3) DETECT MARKERS AND INTERPOLATE CHARUCO CORNERS
    # ---------------------------------------------------
    # Load the board image (like reading from file).
    # But here we already have it in memory, so let's just copy it.
    img = board_img.copy()

    # Create default ArUco detection parameters
    params = cv2.aruco.DetectorParameters()

    # Detect markers
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=params)

    if marker_ids is not None and len(marker_ids) > 0:
        print(f"[DEBUG] Detected {len(marker_ids)} markers with IDs:", marker_ids.ravel())

        # Interpolate Charuco corners
        charuco_retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)

        if charuco_retval > 0:
            print(f"[SUCCESS] Found {charuco_retval} Charuco corners!")
            # Draw the detected markers for debugging
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
            # Optionally draw Charuco corners
            for corner in charuco_corners:
                corner_int = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(debug_img, corner_int, 5, (0, 255, 0), -1)

            cv2.imshow("Detected Charuco Corners", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("[WARN] Charuco interpolation returned 0 corners!")
    else:
        print("[WARN] No ArUco markers detected at all.")

if __name__ == "__main__":
    main()
