import numpy as np
import cv2
from matplotlib import pyplot as plt
from planarH import compositeH

left = cv2.imread('../data/pano_left.jpg')
right = cv2.imread('../data/pano_right.jpg')
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

pad_height = int(right.shape[0]*0)
pad_width = int(right.shape[1]*0.41)
right_padded = np.zeros((right.shape[0] + 2*pad_height, right.shape[1] + pad_width, right.shape[2]), dtype = np.uint8)
right_padded[pad_height:(right.shape[0] + pad_height), pad_width:(right.shape[1] + pad_width),:] = right

right_gray = cv2.cvtColor(right_padded, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create()
locs1, descs1 = orb.detectAndCompute(left_gray,None)
locs2, descs2 = orb.detectAndCompute(right_gray,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descs1, descs2)
matchedLocs1 = np.array([locs1[match.queryIdx].pt for match in matches])
matchedLocs2 = np.array([locs2[match.trainIdx].pt for match in matches])

# res = cv2.drawMatches(left, locs1, right_padded, locs2, matches,None,flags=2)

# H2to1, _ = cv2.findHomography(matchedLocs2, matchedLocs1, cv2.RANSAC)
H1to2, _ = cv2.findHomography(matchedLocs1, matchedLocs2, cv2.RANSAC)

# comp2to1 = compositeH(H2to1, right, left)
comp1to2 = compositeH(H1to2, left, right_padded)
# warped = cv2.warpPerspective(right, H2to1, (left.shape[1],left.shape[0]))
# composite_img = cv2.bitwise_or(left, warped)

plt.imshow(comp1to2)
plt.show()