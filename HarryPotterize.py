import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt
from helper import plotMatches
#Import necessary functions

from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q2.2.4
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# cover_rgb = cv2.cvtColor(hp_cover, cv2.COLOR_BGR2RGB)
# plt.imshow(cover_rgb)
# plt.show()

matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)

#different coordinate system
locs1[:,[0,1]] = locs1[:,[1,0]]
locs2[:,[0,1]] = locs2[:,[1,0]]

matchedLocs1 = np.array([locs1[ind] for ind in matches[:,0]])
matchedLocs2 = np.array([locs2[ind] for ind in matches[:,1]])
bestH2to1, inliers = computeH_ransac(matchedLocs1, matchedLocs2, opts)

hp_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

combined = compositeH(bestH2to1, hp_resized, cv_desk)

cover_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
plt.imshow(cover_rgb)
plt.show()