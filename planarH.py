import numpy as np
import cv2
import random


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	arrX1 = np.asarray(x1)
	arrX2 = np.asarray(x2)
	assert arrX1.shape[0] >= 4, "Need 4 points to compute planar homography"
	assert arrX1.shape == arrX2.shape, "Arrays need to be same size to compute planar homography"
	
	A = np.ndarray((0,9))
	for pair in zip(arrX1, arrX2):
		u, v = pair[0] #u, v
		x, y = pair[1] #x, y
		A = np.vstack((A, [x,y,1,0,0,0,-x*u,-y*u,-u],[0,0,0,x,y,1,-x*v,-y*v,-v]))
	
	_, _, vh = np.linalg.svd(A)
	H2to1 = np.reshape(vh[8,:],(3,3))

	# #Asserting that this is correct, for debugging
	# #start by making x2 homogeous
	# homX2 = np.hstack((arrX2, np.ones((arrX2.shape[0],1))))
	# #apply homography
	# maybe1 = np.asarray([np.matmul(H2to1, homX2[row]) for row in range(homX2.shape[0])])
	# print("IN COMPUTEH: maybe1:\n{}".format(maybe1))
	# maybe1 = np.array([maybe1[row,:2]/maybe1[row,2] for row in range(maybe1.shape[0])])
	# print("IN COMPUTEH: locs1:\n{}".format(arrX1))
	# print("IN COMPUTEH: maybe1:\n{}".format(maybe1))

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	arrX1 = np.asarray(x1)
	arrX2 = np.asarray(x2)

	assert arrX1.shape[0] >= 4, "Need 4 points to compute planar homography"
	assert arrX1.shape == arrX2.shape, "Arrays need to be same size to compute planar homography"

	#concat points to make operations single line
	cat = np.hstack((arrX1, arrX2))

	#find the mean for both x and y
	xmean1, ymean1, xmean2, ymean2 = np.mean(cat, axis=0)

	#move centers by mean amount
	centerMoved =  cat - [xmean1, ymean1, xmean2, ymean2]

	#find scale factor, which is furthest distance from origin of mean-moved points
	sx1, sy1, sx2, sy2 = np.max(np.abs(centerMoved), axis = 0)

	#construct full transformation matrices that will move center and scale appropriately
	T1 = np.asarray([[1/sx1, 0,     -xmean1/sx1],
					 [0,     1/sy1, -ymean1/sy1],
					 [0,     0,     1]
					])
	T2 = np.asarray([[1/sx2, 0,     -xmean2/sx2],
					 [0,     1/sy2, -ymean2/sy2],
					 [0,     0,     1]
					])

	#make homogeneous coordinates out of originals
	homX1 = np.hstack((arrX1, np.ones((arrX1.shape[0],1))))
	homX2 = np.hstack((arrX2, np.ones((arrX2.shape[0],1))))

	#normalized coordinates with transformation matrices
	norm1 = np.asarray([np.matmul(T1, homX1[row])[:2] for row in range(homX1.shape[0])])
	norm2 = np.asarray([np.matmul(T2, homX2[row])[:2] for row in range(homX2.shape[0])])

	#get H of the normalized coordinates
	hNorm = computeH(norm1, norm2)

	#denormalize using scaling matrices
	H2to1 = np.matmul(np.matmul(np.linalg.inv(T1), hNorm), T2)

	#Asserting that this is correct, for debugging
	#apply homography
	# maybe1 = np.asarray([np.matmul(H2to1, homX2[row]) for row in range(homX2.shape[0])])
	# print("IN COMPUTEH_NORM: maybe1:\n{}".format(maybe1))
	# maybe1 = np.array([maybe1[row,:2]/maybe1[row,2] for row in range(maybe1.shape[0])])
	# print("IN COMPUTEH_NORM: locs1:\n{}".format(arrX1))
	# print("IN COMPUTEH_NORM: maybe1:\n{}".format(maybe1))

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points

	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	
	assert locs1.shape == locs2.shape, "Locations need to be same shape for RANSAC"

	currBest = -1 #best number of inliers so far
	inliers = [] #placeholder
	bestH2to1 = [] #placeholder
	for i in range(max_iters):
		inds = random.sample(range(locs1.shape[0]), 4)
		l1Samp = np.array([locs1[ind] for ind in inds])
		l2Samp = np.array([locs2[ind] for ind in inds])
		thisH = computeH_norm(l1Samp, l2Samp)

		#apply to all points, see which results are inliers
		
		#make locs2 homogeneous
		homX2 = np.hstack((locs2, np.ones((locs2.shape[0],1))))

		#apply homography
		maybe1 = np.asarray([np.matmul(thisH, homX2[row]) for row in range(homX2.shape[0])])

		#rescale homography so 3rd position is 1
		maybe1 = np.array([maybe1[row,:2]/maybe1[row,2] for row in range(maybe1.shape[0])])

		#caluclate euclidean distances of projected points to real points
		dists = np.linalg.norm(locs1 - maybe1, axis=1)

		theseInliers = (dists < inlier_tol) + 0 #to make 0 or 1

		if sum(theseInliers) > currBest:
			#save this configuration and the inlier indices
			currBest = sum(theseInliers)
			inliers = theseInliers
			bestH2to1 = thisH
	
	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	#Unclear function definiton here, assuming the tamplate passed in
	#is unwarped but the correct size
	template_warped = cv2.warpPerspective(template, H2to1, (img.shape[1],img.shape[0]))

	#All white template
	template_blank = 255*np.ones(template.shape[:2], dtype=np.uint8)
	#warp
	template_blank_warped = cv2.warpPerspective(template_blank, H2to1, (img.shape[1],img.shape[0]))
	#invert
	template_blank_inv = cv2.bitwise_not(template_blank_warped)
	#apply inverted template to mask img
	img_masked = cv2.bitwise_and(img, img, mask=template_blank_inv)
	#combine with warped template
	composite_img = cv2.bitwise_or(img_masked, template_warped)

	return composite_img


