import numpy as np
import cv2
from matplotlib import pyplot as plt
from planarH import compositeH

left = cv2.imread('../data/my_pano_left.jpg')
right = cv2.imread('../data/my_pano_right.jpg')
left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

#pad right image with some black space on left
pad_height = int(right.shape[0]*0)
pad_width = int(right.shape[1]*0.41)
right_padded = np.zeros((right.shape[0] + 2*pad_height, right.shape[1] + pad_width, right.shape[2]), dtype = np.uint8)
right_padded[pad_height:(right.shape[0] + pad_height), pad_width:(right.shape[1] + pad_width),:] = right

right_gray = cv2.cvtColor(right_padded, cv2.COLOR_BGR2GRAY)

#find features in left and padded right
orb = cv2.ORB_create(nfeatures = 5000)
locs1, descs1 = orb.detectAndCompute(left_gray,None)
locs2, descs2 = orb.detectAndCompute(right_gray,None)

#match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descs1, descs2)
sortedMatches = sorted(matches, key = lambda x:x.distance)#sort best at top
matchedLocs1 = np.array([locs1[match.queryIdx].pt for match in sortedMatches[:500]])#take 500 best matches
matchedLocs2 = np.array([locs2[match.trainIdx].pt for match in sortedMatches[:500]])

#find homography that translates left to padded right
H1to2, _ = cv2.findHomography(matchedLocs1, matchedLocs2, cv2.RANSAC, 20.0)

#add warped left to padded right 
comp1to2 = compositeH(H1to2, left, right_padded)

#show!
plt.imshow(comp1to2)
plt.show()