import os
import sys
import numpy as np
import pandas as pd

from pymatgen import Composition
from pymatgen.core.periodic_table import get_el_sp, Element, Specie
from xenonpy.descriptor import Compositions as comp
from xenonpy.datatools import preset

import macro

# get atomic number using samples
# return atomic numbers used in the database and atom lists
def get_atomic_lists_and_numbers(compositions):

	# get atom number
	atom_lists = np.empty(118,dtype=object)
	atoms_duplicated = []
	atomic_numbers_duplicated = []
	for line in compositions:
		if line == 'nan':
			line = 'NaN'
		atom_list = Composition(line[0])
		#atom_list = Composition(line)
		for element in atom_list:
			atomic_numbers_duplicated.append(element.Z-1)
			atom_lists[element.Z-1] = element.symbol

	# reduce duplicate
	atomic_numbers = np.array(list(set(atomic_numbers_duplicated)))
	atomic_numbers = np.sort(atomic_numbers, axis=0)

	return atom_lists, atomic_numbers

# load materials dataset using xenonpy and pandas
def download_file():

	data = pd.read_csv(macro._INORGANIC_DATA,delimiter=',',engine="python")

	composition_lists = data['Composition'].map(lambda x: Composition(x))
	composition_lists = [x.as_dict() for x in composition_lists]
	features = np.array(comp().transform(composition_lists))
	feature_num = np.array(preset.elements_completed.head(1)).shape[1]
	features, _ = np.split(features,[feature_num], axis=1)

	data = np.array(data)
	ids, compositions, labels = np.split(data,[1,2],axis=1)
	compositions = np.array(compositions,dtype=str)
	labels = np.array(labels,dtype=np.float32)

	return compositions, labels, features

# get data from .csv file and compute features
def get_data():

	# load materials project data
	compositions, labels, features = download_file()

	# Get atomic numbers and atom lists using this dataset
	atom_lists, atomic_numbers = get_atomic_lists_and_numbers(compositions)

	# By checking this list, we can reconstruct the composition (i-th variable correponds to i-th atom)
	save_data = pd.DataFrame(atom_lists).dropna()
	save_data.to_csv(macro._ATOM_LIST, index=False, header=False)

	return labels, compositions, features, atomic_numbers
