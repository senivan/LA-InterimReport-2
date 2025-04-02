#!/usr/bin/env python3
# In this file, we will use calibrated cameras from the calibration.py to draw the 
# epipolar lines on the images. and then calculte disparity map using the epipolar lines.
# All of this will be done using OpenCV and minimal use of numpy.
import cv2
import numpy as np
import calibration

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



if __name__ == "__main__":
    K, extrinsics = calibration.multi_image_calibration("prod_calib", (8, 6), 0.025)
    
    # -- Load stereo images -- #
    imgL = cv2.imread("left.png", cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)
    
    # Ensure images are loaded
    if imgL is None or imgR is None:
        print("Error: Could not load images.")
        exit()

    # Convert to color for visualization
    imgL_lines = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    imgR_lines = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    # Get extrinsics
    R, t = extrinsics[0]
    
    # Detect ORB keypoints and match them
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(imgL, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imgR, None)

    # Use brute-force matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Extract matching points
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Compute fundamental matrix
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Compute epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # Draw epipolar lines
    def draw_lines(img, lines, pts):
        for line, pt in zip(lines, pts):
            a, b, c = line
            x0, y0 = 0, int(-c / b)  # Left boundary
            x1, y1 = img.shape[1], int(-(c + a * img.shape[1]) / b)  # Right boundary
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.circle(img, tuple(map(int, pt)), 5, (0, 255, 0), -1)  # Draw corresponding point

    draw_lines(imgL_lines, lines1, pts1)
    draw_lines(imgR_lines, lines2, pts2)

    # Show the images
    plt.subplot(121), plt.imshow(imgL_lines)
    plt.title('Left Image with Epipolar Lines'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(imgR_lines)
    plt.title('Right Image with Epipolar Lines'), plt.xticks([]), plt.yticks([])
    plt.show()

    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=35)
    disparity_map = stereo.compute(imgL, imgR)

    # Normalize disparity for visualization
    disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.uint8(disparity_map)

    # --- Convert Disparity to Depth ---
    # f = 726  # Example focal length in pixels (from intrinsic matrix)
    # B = 0.1  # Example baseline distance in meters (distance between cameras)
    f = 3740  # Focal length from intrinsic matrix
    B = 0.16  # Baseline from extrinsics (translation vector)
    print("Focal Length (f):", f)
    print("Baseline (B):", B)
    # Avoid division by zero
    disparity_map = np.where(disparity_map == 0, 1, disparity_map)  # Avoid zero disparity
    depth_map = (f * B) / disparity_map

    # Normalize depth for visualization
    depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)

    # --- Display Disparity & Depth Map ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(disparity_map, cmap='gray')
    plt.title("Disparity Map")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap='jet')
    plt.title("Depth Map")
    plt.axis("off")

    plt.show()