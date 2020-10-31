import cv2
import numpy as np 
import math
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from create_voronoi import colorizedVoronoi

def dot( img, points, size, negative = False ):

	color = 255 if not negative else 0

	result = np.zeros( img.shape[:2], dtype = np.uint8 ) + color
	for x,y in points:
		cv2.circle(result, (int(round(x)), int(round(y))), 3, 255 - color, -1 ) 

	return result

def dot_color( img, points, size, negative = False ):

	color = 255 if not negative else 0

	colorList = [ i for i in range(len(points)) ]

	# vor = colorizedVoronoi( points, colorList, img.shape )

	result = np.zeros( img.shape, dtype = np.uint8 ) + color

	img = np.float64( img )

	for id_ in colorList:
		try:
			# b = int(np.sum( img[ idx ][0] ) / len( idx )) 
			# g = int(np.sum( img[ idx ][1] ) / len( idx ))
			# r = int(np.sum( img[ idx ][2] ) / len( idx ))

			# result[idx] = sum_
			x,y = points[ id_ ]
			b = img[ int(y), int(x), 0 ]
			g = img[ int(y), int(x), 1 ]
			r = img[ int(y), int(x), 2 ]
			cv2.circle(result, (int(round(x)), int(round(y))), size, (b,g,r), -1 ) 
		except IndexError:
			pass

	return result

def line( img, points, ):

	vor = Voronoi(points)

	result = np.zeros( img.shape[:2], dtype = np.uint8 ) + 255
	
	for id1, id2 in vor.ridge_points:

		p1 = tuple(np.int32(points[id1]))
		p2 = tuple(np.int32(points[id2]))

		cv2.line( result, p1, p2, 0, 1)

	return result

def line_color( img, points, line_thick = 1 ):

	vor = Voronoi(points)

	mask = np.zeros( img.shape[:2], dtype = np.uint8 )

	blur = cv2.GaussianBlur(img,(11,11),0)
	
	for id1, id2 in vor.ridge_points:

		p1 = tuple(np.int32(points[id1]))
		p2 = tuple(np.int32(points[id2]))

		center = ( (p1[0] + p2[0])/2, (p1[1] + p2[1])/2 )

		# b1 = int(img[ int(p1[1]), int(p1[0]), 0 ])
		# g1 = int(img[ int(p1[1]), int(p1[0]), 1 ])
		# r1 = int(img[ int(p1[1]), int(p1[0]), 2 ])

		# b2 = int(img[ int(p2[1]), int(p2[0]), 0 ])
		# g2 = int(img[ int(p2[1]), int(p2[0]), 1 ])
		# r2 = int(img[ int(p2[1]), int(p2[0]), 2 ])

		# print b1,g1,r1, p1, center
		cv2.line( mask, p1, center, 1, line_thick)
		# cv2.circle(result, p1, 2, (b1,g1,r1), -1 )

		cv2.line( mask, p2, center, 1, line_thick)
		# cv2.circle(result, p2, 2, (b2,g2,r2), -1 )

	result = cv2.bitwise_and( blur, blur, mask = mask )
	neg_mask = 1 - mask
	result[:,:,0] += np.ones( img.shape[:2], dtype = np.uint8 )*255*neg_mask
	result[:,:,1] += np.ones( img.shape[:2], dtype = np.uint8 )*255*neg_mask
	result[:,:,2] += np.ones( img.shape[:2], dtype = np.uint8 )*255*neg_mask

	return result

# def line( img, points, ):
	
# 	distMat = cdist( points, points, metric='euclidean' )

# 	maxDist = distMat.max( )

# 	np.fill_diagonal(distMat, maxDist + 1)

# 	result = np.zeros( img.shape[:2], dtype = np.uint8 ) + 255

# 	i = 0
# 	n = 0
# 	while n != len(points) - 1:
# 		# print n
# 		minIdx = np.argmin( distMat[i] )

# 		p1 = tuple(np.int32(points[ i ]))
# 		p2 = tuple(np.int32(points[ minIdx ]))

# 		if distMat[i, minIdx] < 30:
# 			cv2.line( result, p1, p2, 0, 1)

# 		distMat[:,i] = maxDist + 1

# 		i = minIdx
# 		n+=1

# 	return result

# def line( img, points, ):
	
# 	distMat = cdist( points, points, metric='euclidean' )

# 	# maxDist = distMat.max( )

# 	# np.fill_diagonal(distMat, maxDist + 1)

# 	result = np.zeros( img.shape[:2], dtype = np.uint8 ) + 255

# 	# i = 0
# 	# n = 0
# 	# while n != len(points) - 1:
# 	# 	# print n
# 	# 	minIdx = np.argmin( distMat[i] )

# 	# 	p1 = tuple(np.int32(points[ i ]))
# 	# 	p2 = tuple(np.int32(points[ minIdx ]))

# 	# 	if distMat[i, minIdx] < 30:
# 	# 		cv2.line( result, p1, p2, 0, 1)

# 	# 	distMat[:,i] = maxDist + 1

# 	# 	i = minIdx
# 	# 	n+=1

# 	# return result
# 	import tsp
# 	t = tsp.tsp([(0,0), (0,1), (1,0), (1,1)])

# 	D = [ distMat[i, :i].tolist() if i != 0 else [] for i in range( len( points ) ) ]



# 	path = tsp.tsp(points)[1]

# 	for i1, i2 in zip( path[:-1], path[1:] ):

# 		x1 = int(round(points[i1, 0]))
# 		y1 = int(round(points[i1, 1]))

# 		x2 = int(round(points[i2, 0]))
# 		y2 = int(round(points[i2, 1]))

# 		cv2.line( result, (x1,y1), (x2,y2), 0, 1)

# 	return result