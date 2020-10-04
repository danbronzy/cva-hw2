import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

#nice little loop to save the picutres as I go
for rat in np.linspace(0.5, 0.9, 3):
    for sig in np.linspace(.1, .2, 3):
        opts.ratio = rat
        opts.sigma = sig
        print("Ratio: {} - Sig: {}".format(opts.ratio, opts.sigma))
        matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
        print("\tNumber: {}".format(len(matches)))
        plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

