from create_voronoi import colorizedVoronoi
from visualize import dot, line, dot_color
import cv2
import numpy as np
import time
import random
import os
import math
import time
import multiprocessing as mp

n = 0

def lloyd_update( points, weightImg ):

	colorList = [ i for i in range(len(points)) ]
	t0 = time.time()
	voronoi = colorizedVoronoi( points, colorList, weightImg.shape )
	t1 = time.time()
	print( "Finish voronoi" )
	newPoints = []
	# for indx in colorList:	
	# 	centroid = findCentroid( voronoi, indx, weightImg.copy() )
	# 	# print points[ indx ], 'Befor'
	# 	# print (x,y), 'After'
	# 	# print '--'
	# 	if centroid is None:
	# 		continue
	# 	# print centroid
	# 	newPoints.append( centroid )
	newPoints = findCentroidParallelize( voronoi, len(colorList), weightImg.copy() )
	t2 = time.time()
	print( "Find voronoid: {}, find centroid: {}".format( t1 - t0, t2 - t1 ) )
	return np.vstack( newPoints )

def findCentroid_slow(voronoi, id, pdf_img):
	indx = np.where( voronoi == id )
	m  = np.sum( pdf_img[indx] )
	mx = np.sum( indx[1]*pdf_img[indx] )
	my = np.sum( indx[0]*pdf_img[indx] )

	if m == 0:
		return 

	y = (my / m).astype(int)
	x = (mx / m).astype(int)

	return x,y

def findCentroid( voronoi, indx, weightImg ):
	weightImg = weightImg.copy()
	weightImg[ voronoi != indx ] = 0

	m = cv2.moments( weightImg )

	if m['m00'] != 0:
		x = m['m10'] / float(m['m00'])
		y = m['m01'] / float(m['m00'])

	else:
		return

	if not( 0<=x<=weightImg.shape[1] or 0<=y<=weightImg.shape[0]):
		print( x,y )

	return x, y

def findCentroidParallelize( voronoi, nPoint, weightImg ):

	arguments = [ (voronoi, i, weightImg) for i in range( nPoint )]

	with mp.Pool( 10 ) as pool:
		results = pool.starmap( findCentroid, arguments )

	results = [ r for r in results if r is not None ]

	return results

def generate_point(numPoint, img, cutoff = 220):
	cutoff = cutoff if cutoff < 255 else 254
	genPoint = []
	img_temp = img.copy()
	shape = img.shape
	max_itr = shape[0]*shape[1]
	for i in range(numPoint):
		y = (shape[0]-1) * random.random()
		x = (shape[1]-1) * random.random()
		count = 0
		while img_temp[int(y),int(x)] < cutoff * random.random():
			y = (shape[0]-1) * random.random()
			x = (shape[1]-1) * random.random()
			count += 1

		img_temp[int(y),int(x)] = 255
		genPoint.append( [ x, y] )
	return np.vstack(genPoint)

if __name__ == '__main__':
	fn = 'pass.jpg'
	PATH = fn.split('.')[0]
	try:
		os.mkdir(PATH)
	except:
		pass

	img = cv2.imread( fn )
	img = cv2.resize(img, (0,0), fx = 1.0, fy = 1.0)

	weightImg = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

	#	We want negative image
	weightImg = 255 - weightImg

	ret, weightImg = cv2.threshold( weightImg, 25, 255, cv2.THRESH_TOZERO )

	bp = 25
	wp = 255
	lookup = [ int((wp*(i - bp)) / (wp - bp)) if bp<=i<=wp else 0 for i in range(255) ]
	lookup = np.array( lookup, dtype = np.uint8 )
	print(lookup)

	weightImg = lookup[ weightImg ]

	cv2.imshow( 'eee', weightImg )
	cv2.waitKey( 0 )
	cv2.destroyAllWindows(  )

	points = generate_point( 10000, weightImg, cutoff = 100 )

	for i in range( 100 ):

		#	Let save it every 10 iterations
		if i % 10 == 0:
			result = dot_color( img, points, 3 )

			np.save( PATH + '/' + fn.split( '.' )[0] + '_{}'.format( i ) + '.npy', points )
			cv2.imwrite( PATH + '/' + fn.split( '.' )[0] + '_{}'.format( i ) + '.png', result )

		print( i + 1 )
		t0 = time.time()
		points = lloyd_update( points, weightImg )
		print( len(points) )
		print( "Time 1 iteration: ", time.time() - t0 )
		print( "Drop : ", n )
		print( '----' )

	result = dot_color( img, points, 3 )
	np.save( PATH + '/' + fn.split( '.' )[0] + '_{}'.format( i ) + '.npy', points )
	cv2.imwrite( PATH + '/' + fn.split( '.' )[0] + '_{}'.format( i ) + '.png', result )
	cv2.imshow( 'result', result )
	cv2.waitKey(0)
	cv2.destroyAllWindows( )