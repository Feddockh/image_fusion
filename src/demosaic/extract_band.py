#!/usr/bin/env python3

import cv2
import numpy as np
import argparse

def demosaic_image(input_path, output_path):
    # Load the TIFF image
    cv_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if cv_image is None:
        print(f"Failed to load image from {input_path}")
        return

    # Optionally, if the image dimensions are not divisible by 5, crop the image
    height, width = cv_image.shape
    print(cv_image.shape)
    if height % 5 != 0 or width % 5 != 0:
        new_height = (height // 5) * 5
        new_width = (width // 5) * 5
        print(f"Image dimensions ({height}x{width}) not divisible by 5, cropping to {new_height}x{new_width}.")
        cv_image = cv_image[:new_height, :new_width]

    # Extract the top-left pixel from each 5x5 block
    demosaicked_image = cv_image[0::5, 0::5]

    # Save the resulting image as a PNG
    cv2.imwrite(output_path, demosaicked_image)
    print(f"Saved demosaicked image to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Demosaic a TIFF image and save as PNG.")
    parser.add_argument('--input_path', type=str, default='data/1739373925_363698530_ximea.tif', help="Path to the input TIFF image.")
    parser.add_argument('--output_path', type=str, default='output.png', help="Path to save the output PNG image.")
    args = parser.parse_args()

    demosaic_image(args.input_path, args.output_path)

if __name__ == '__main__':
    main()