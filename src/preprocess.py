import os
import sys
import numpy as np
from pymatgen import Composition

import macro

# normalization function min is to set 0, max is to set 1
# return normalized data, and max and min of each variable
def normalize_for_train(data):

	data_max = np.max(data, axis=0)
	data_min = np.min(data, axis=0)

	normalized_data = (data-data_min)/(data_max-data_min+macro._EPS)

	return normalized_data, data_max, data_min


# paired shuffle function
def paired_shuffle(data,labels):

	# concatenate
	zips = np.zeros((data.shape[0],data.shape[1]+labels.shape[1]+1), dtype=np.float32)
	for i in range(len(data)):
		zips[i] = np.hstack((labels[i],i,data[i]))

	# shuffle
	np.random.shuffle(zips)

	# separate
	slabels = zips[:,0:1]
	sidx = zips[:,1:2]
	sdata = zips[:,2:]

	return sdata, slabels, sidx


# return bag-of-atoms vectors
def compute_bag_of_atom_vector(compositions, atomic_numbers):

	i = 0
	boa_vectors = np.zeros((len(compositions),len(atomic_numbers)),dtype=np.float32)
	for line in compositions:
		if line == 'nan':
			line = 'NaN'
		# get atom information from composition
		atoms = Composition(line[0])
		for element in atoms:
			# get ratio of each atom of composition
			ratio = atoms.get_atomic_fraction(element)
			if element.Z-1 < len(atomic_numbers) and \
			element.Z-1 == atomic_numbers[element.Z-1]:
					idx = element.Z-1
			else:
				for j in range(len(atomic_numbers)-1,-1,-1):
					if element.Z-1 == atomic_numbers[j]:
						idx = j
						break
			boa_vectors[i][idx] = ratio
		i=i+1
	return boa_vectors
