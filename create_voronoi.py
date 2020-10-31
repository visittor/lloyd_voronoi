import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Voronoi
from collections import namedtuple
import math

def voronoi_finite_polygons_2d(vor, radius=None):
	"""
	Reconstruct infinite voronoi regions in a 2D diagram to finite
	regions.

	Parameters
	----------
	vor : Voronoi
		Input diagram
	radius : float, optional
		Distance to 'points at infinity'.

	Returns
	-------
	regions : list of tuples
		Indices of vertices in each revised Voronoi regions.
	vertices : list of tuples
		Coordinates for revised Voronoi vertices. Same as coordinates
		of input vertices, with 'points at infinity' appended to the
		end.

	"""

	if vor.points.shape[1] != 2:
		raise ValueError("Requires 2D input")

	new_regions = []
	new_vertices = vor.vertices.tolist()

	center = vor.points.mean(axis=0)
	if radius is None:
		radius = vor.points.ptp().max()

	# Construct a map containing all ridges for a given point
	all_ridges = {}
	for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
		all_ridges.setdefault(p1, []).append((p2, v1, v2))
		all_ridges.setdefault(p2, []).append((p1, v1, v2))

	# Reconstruct infinite regions
	for p1, region in enumerate(vor.point_region):
		if not p1 in all_ridges:
			continue
		vertices = vor.regions[region]

		if all(v >= 0 for v in vertices):
			# finite region
			new_regions.append(vertices)
			continue

		# reconstruct a non-finite region
		ridges = all_ridges[p1]
		new_region = [v for v in vertices if v >= 0]

		for p2, v1, v2 in ridges:
			if v2 < 0:
				v1, v2 = v2, v1
			if v1 >= 0:
				# finite ridge: already in the region
				continue

			# Compute the missing endpoint of an infinite ridge

			t = vor.points[p2] - vor.points[p1] # tangent
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])  # normal

			midpoint = vor.points[[p1, p2]].mean(axis=0)
			# print( midpoint, center, vor.points )
			direction = np.sign(np.dot(midpoint - center, n)) * n
			far_point = vor.vertices[v2] + direction * radius

			new_region.append(len(new_vertices))
			new_vertices.append(far_point.tolist())

		# sort region counterclockwise
		vs = np.asarray([new_vertices[v] for v in new_region])
		c = vs.mean(axis=0)
		angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
		new_region = np.array(new_region)[np.argsort(angles)]

		# finish
		new_regions.append(new_region.tolist())

	return new_regions, np.asarray(new_vertices)

def voronoiToPolygon( voronoi : Voronoi, maxDist, center ):
	'''
		Voronoi from scipy contain vertices at infinity. This function
		will find polygon of the voronoi in finite distance by giving max
		radius it can go.
	'''
	regions = {}

	RidgeInfo = namedtuple( "RidgeInfo", ["otherIdx", "v1Idx", "v2Idx"] )

	for (p1Idx, p2Idx), (v1Idx, v2Idx) in zip( voronoi.ridge_points, voronoi.ridge_vertices ):

		regions.setdefault( p1Idx, [] ).append( RidgeInfo( p2Idx, v1Idx, v2Idx ) )
		regions.setdefault( p2Idx, [] ).append( RidgeInfo( p1Idx, v1Idx, v2Idx ) )

	vertices = voronoi.vertices

	regionsList =  list(regions.items())
	regionsList.sort( key = lambda x : x[0] )

	outVertices = vertices.tolist()
	outRegions = [  ]

	for pIdx, ridgeInfos in regionsList:
		outRegion = set()

		for ridgeInfo in ridgeInfos:

			otherIdx = ridgeInfo.otherIdx
			v1Idx = ridgeInfo.v1Idx
			v2Idx = ridgeInfo.v2Idx

			#	these vertices in finite space don't have to process
			if v1Idx >= 0 and v2Idx >= 0:
				outRegion.add( v1Idx )
				outRegion.add( v2Idx )
				continue

			assert( v1Idx >= 0 or v2Idx >= 0 ), "Both vertices cannot be at inifinite at the same time"

			if v1Idx < 0:
				v1Idx, v2Idx = v2Idx, v1Idx

			p1 = voronoi.points[ pIdx ]
			p2 = voronoi.points[ otherIdx ]

			v1 = voronoi.vertices[ v1Idx ]

			#	find direction of the ridge to find a finite end point
			midPoint = (p1 + p2) / 2
			normal = np.array( [p1[1]-p2[1], p2[0]-p1[0]] )
			normal /= np.linalg.norm( normal )
			direction = np.sign(np.dot(midPoint - np.array(center), normal)) * normal

			#	calculate endpoint
			v2 = v1 + direction*maxDist

			outVertices.append( v2 )
			outRegion.add( v1Idx )
			outRegion.add( len(outVertices) - 1 )

		#	sort vertices by angle repect to center of vertices.
		outRegion = list( outRegion )
		vList = np.array( [ outVertices[idx] for idx in outRegion ] )
		cen = vList.mean( axis = 0 )
		angs = np.arctan2( vList[:,1] - cen[1], vList[:,0] - cen[0] )
		outRegion = np.array(outRegion)[ np.argsort( angs ) ]

		outRegions.append( outRegion.tolist() )

	return outRegions, np.vstack(outVertices) 

def colorizedVoronoi( points, colorList, imgShape ):
	vor = Voronoi(points)

	# radius = math.sqrt(imgShape[0]**2 + imgShape[1]**2)*2
	radius = 25
	# regions, vertices = voronoi_finite_polygons_2d(vor, radius = radius)
	regions, vertices = voronoiToPolygon( vor, radius, (imgShape[0]/2, imgShape[1]/2) )

	img = np.zeros( imgShape, dtype = np.uint8 )
	voronoi = np.zeros( imgShape[:] )

	# colorize
	for region, color, p in zip(regions, colorList, points ):

		polygon = np.around(vertices[region]).astype( np.int32 )

		#	NOTE: not sure if opencv work with float image
		#			This is work around by create mask and then
		#			use mark to assign value
		img = np.zeros( imgShape[:2], dtype = np.uint8 )
		cv2.fillPoly(img, [polygon], 1)

		voronoi[ img == 1 ] = color

	return voronoi

if __name__ == '__main__':
	# np.random.seed( 0 )
	points = np.random.rand(15, 2) * 440 + 20
	# points = np.array( [ [200, 240], [280, 240], [240, 280], [240, 200] ])

	colorList = np.random.uniform( low = 127, high = 255, size = (15,3) )

	# # compute Voronoi tesselation
	# vor = Voronoi(points)

	# # plot
	# regions, vertices = voronoi_finite_polygons_2d(vor, radius = 480)
	# print "--"
	# print regions
	# print "--"
	# print vertices

	img = colorizedVoronoi( points, colorList, (480, 640, 3) ) / 255

	ax = plt.gca()

	for cen in points:
		ax.add_artist( plt.Circle( tuple(cen), radius=3 ) )
	plt.imshow( img )
	plt.show()