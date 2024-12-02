import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

# Load images
img1 = cv.imread('book.jpg', cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('book_in_scene.jpg', cv.IMREAD_GRAYSCALE)  # trainImage

traces = []  # To store the trace of each homography matrix

for i in range(15):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                         ).reshape(-1, 1, 2)

        # Compute homography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is not None:
            # Calculate trace of the matrix
            M_trace = np.trace(M)
            traces.append(M_trace)
            print(traces)
        else:
            print(f"Iteration {i+1}: Homography computation failed.")
            # Append NaN to maintain alignment
            traces.append(float('nan'))

    else:
        print(
            f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        matchesMask = None
else:
    print("Code doesn't work")
# Draw matches
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=None,  # change to mask if inliers are needed
                   flags=2)

# Determine the most stable trace
stable_idx = np.nanargmin(
    np.abs(np.array(traces) - np.nanmean(traces)))
print(
    f"The most stable iteration is {stable_idx + 1} with trace value: {traces[stable_idx]:.4f}")


# Plot the traces
plt.figure(figsize=(12, 6))
plt.plot(range(1, 16), traces, marker='o',
         label='Trace of Homography Matrix')
plt.axhline(np.nanmean(traces), color='r',
            linestyle='--', label='Mean Trace')
plt.title("Homography Matrix Trace Over 15 Iterations")
plt.xlabel("Iteration")
plt.ylabel("Trace Value")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the stability of the traces
print("Traces from all iterations:")
print(traces)


img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
