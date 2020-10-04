import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import scipy.ndimage
from matplotlib import pyplot as plt

opts = get_opts()

#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
counts = np.zeros(36)

for i in range(36):
	print("Rotation: {} degrees".format(i * 10))
	#Rotate Image
	rot = 10 * i #radians
	cv_cover_rot = scipy.ndimage.rotate(cv_cover, rot)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, cv_cover_rot, opts)

	#plot 3 different cases
	if i < 3: #i=0,1,2
		plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)
	
	#Update histogram
	counts[i] = len(matches)

#Display histogram
plt.bar(10*np.asarray(range(36)), counts, align='edge')
plt.xlabel("Rotation [\N{DEGREE SIGN}]")
plt.ylabel("Number of matches")
plt.show()
