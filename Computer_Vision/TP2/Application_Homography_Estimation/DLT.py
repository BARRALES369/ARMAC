import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Constants
MIN_MATCH_COUNT = 10

# Load images
img1 = cv.imread('book.jpg', cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('book_in_scene.jpg', cv.IMREAD_GRAYSCALE)  # trainImage

# SIFT feature detection
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Check if there are enough matches
if len(good_matches) > MIN_MATCH_COUNT:
    # Get the coordinates of the matched keypoints
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    def dlt(src_pts, dst_pts):
        """
        Basic Direct Linear Transform (DLT) for homography estimation.
        Args:
            src_pts: Source points (Nx2 numpy array)
            dst_pts: Destination points (Nx2 numpy array)
        Returns:
            H: Homography matrix (3x3 numpy array)
        """
        assert src_pts.shape == dst_pts.shape, "Point arrays must have the same shape."

        # Number of points
        n = src_pts.shape[0]

        # Build the matrix A
        A = []
        for i in range(n):
            x, y = src_pts[i]
            x_prime, y_prime = dst_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)

        # Solve for H using SVD
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        # Normalize H to make H[2,2] = 1
        H = H*(1/H[2, 2])

        return H

    # Calculate the homography matrix using DLT
    H = dlt(src_pts, dst_pts)
    # Print the homography matrix
    print("Normalized Homography Matrix:")
    print(H)

    # Transform the query image's corners to the train image's perspective
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, H)

    # Draw the detected object on the train image
    img2_with_polyline = cv.polylines(
        img2.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    # Draw matches
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None, flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2,
                          good_matches, None, **draw_params)

    # Show the result
    plt.imshow(img3, 'gray')
    plt.title("Normalized Homography Result")
    plt.show()

else:
    print(
        f"Not enough matches are found - {len(good_matches)}/{MIN_MATCH_COUNT}")
