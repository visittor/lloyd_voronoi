import random, numpy, math, copy, matplotlib.pyplot as plt

def TSP( cities ):
	# cities = [random.sample(range(100), 2) for x in range(nCity)];
	nCity = len( cities )
	tour = random.sample(range(nCity),nCity);
	for temperature in numpy.logspace(0,7,num=1e+5)[::-1]:
		[i,j] = sorted(random.sample(range(nCity),2));
		newTour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:];

		oldDistances =  sum([ math.sqrt(sum([(cities[tour[(k+1) % nCity]][d] - cities[tour[k % nCity]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])
		newDistances =  sum([ math.sqrt(sum([(cities[newTour[(k+1) % nCity]][d] - cities[newTour[k % nCity]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])
		if math.exp( ( oldDistances - newDistances) / temperature) > random.random():
			tour = copy.copy(newTour);
	# plt.plot([cities[tour[i % nCity]][0] for i in range(16)], [cities[tour[i % nCity]][1] for i in range(16)], 'xb-');
	# plt.show()

	return [cities[tour[i % nCity]][0] for i in range(nCity + 1)], [cities[tour[i % nCity]][1] for i in range(nCity + 1)]