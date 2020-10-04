import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
#Write script for Q3.1

opts = get_opts()

#load videos
frames_panda = loadVid("../data/ar_source.mov")
frames_book = loadVid("../data/book.mov")

#load template image
cv_cover = cv2.imread('../data/cv_cover.jpg')

pandaWidth = int(cv_cover.shape[1]/cv_cover.shape[0] * frames_panda.shape[1])
trim = int(frames_panda.shape[2] - pandaWidth)//2

result = cv2.VideoWriter('../testvid.avi',  
                         cv2.VideoWriter_fourcc(*'MPEG'), 
                         25, (frames_book.shape[2],frames_book.shape[1])) 
                        
for i in range(frames_book.shape[0]):
    print("On frame {}".format(i))
    #get image from video
    book_frame = frames_book[i]

    #get matches and swap columns
    matches, locs1, locs2 = matchPics(book_frame, cv_cover, opts)
    locs1[:,[0,1]] = locs1[:,[1,0]]
    locs2[:,[0,1]] = locs2[:,[1,0]]

    #compute RANSAC
    matchedLocs1 = np.array([locs1[ind] for ind in matches[:,0]])
    matchedLocs2 = np.array([locs2[ind] for ind in matches[:,1]])
    bestH2to1, _ = computeH_ransac(matchedLocs1, matchedLocs2, opts)

    #trim frame and resize
    trimmedFrame = frames_panda[i%frames_panda.shape[0],:,trim:(trim+pandaWidth),:]
    resizedTrim = cv2.resize(trimmedFrame, (cv_cover.shape[1], cv_cover.shape[0]))

    #composite
    combined = compositeH(bestH2to1, resizedTrim, book_frame)

    #write to video
    result.write(combined)

result.release()
