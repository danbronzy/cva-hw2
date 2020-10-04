import numpy as np
import cv2
from time import perf_counter 
import pyqtgraph as pg
#Import necessary functions
from loadVid import loadVid
from planarH import compositeH
from loadVid import loadVid

#read data
cv_cover = cv2.imread('../data/cv_cover.jpg')

frames_panda = loadVid("../data/ar_source.mov")
frames_book = loadVid("../data/book.mov")

#figure out center of panda image
pandaWidth = int(cv_cover.shape[1]/cv_cover.shape[0] * frames_panda.shape[1])
trim = int(frames_panda.shape[2] - pandaWidth)//2

#make gray since features dont need color
cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)

#instantiate ORB and matcher, get features/descriptions from the template, only need to do once
orb = cv2.ORB_create(nfeatures=650, scoreType=cv2.ORB_FAST_SCORE)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
locs2, des2 = orb.detectAndCompute(cover_gray,None)#cv2.keypoint, description

#Conduct frame-by-frame claculations

# Need to use pyqtgraph since matplotlib overhead sucks for realtime applications on my laptop
# With no display, I was generally getting 50-65 FPS for just the calculations.
# With pyqtgraph I'm getting 30-40 FPS for calculations + display
# With matplotlib, I was getting 10-15 FPS for calculations + display

# This is all infra to support fast updating
app = pg.mkQApp()
gv = pg.GraphicsView()
vb = pg.ViewBox()
gv.setCentralItem(vb)
gv.show()
vb.invertY()
thisImg = pg.ImageItem(frames_book[0], axisOrder='row-major')
vb.addItem(thisImg)

#run the loop and display in real time
startTime = perf_counter()
for currFrame in range(frames_book.shape[0]):
    currFrame_gray = cv2.cvtColor(frames_book[currFrame], cv2.COLOR_BGR2GRAY)
    locs1, des1 = orb.detectAndCompute(currFrame_gray, None)

    matches = matcher.match(des1,des2) #dmatch object
    matchedLocs1 = np.array([locs1[match.queryIdx].pt for match in matches])
    matchedLocs2 = np.array([locs2[match.trainIdx].pt for match in matches])
    H2to1, _ = cv2.findHomography(matchedLocs2, matchedLocs1, cv2.RANSAC, 15.0)

    trimmedFrame = frames_panda[currFrame%frames_panda.shape[0],:,trim:(trim+pandaWidth),:]
    resizedTrim = cv2.resize(trimmedFrame, (cv_cover.shape[1], cv_cover.shape[0]))

    comp = compositeH(H2to1, resizedTrim, frames_book[currFrame])
    thisImg.setImage(comp)
    app.processEvents()

endTime = perf_counter()
print("FPS for calculation: {}".format(frames_book.shape[0]/(endTime - startTime)))
print("Average calculation time: {}".format((endTime - startTime)/frames_book.shape[0]))

