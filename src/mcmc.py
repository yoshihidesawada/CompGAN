import os
import sys

import numpy as np

import macro

def metropolis_hastings(valences,chemical_formula,iteration):

	# normalize and rounding
	mean_vect = chemical_formula[chemical_formula>1.0e-6]
	mean_sum = np.sum(mean_vect)
	mean_vect = mean_vect/mean_sum
	mean_vect = np.round(mean_vect,decimals=macro._ROUND)

	# if composition includes only one atom, return
	if len(mean_vect) == 1:
		return mean_vect

	cnt = []
	for i in range(len(valences)):
		cnt.append(len(valences[i]))
	cnt = np.array(cnt,dtype=np.int32)

	buf = np.zeros(cnt.max(),dtype=np.int32)
	for i in range(len(cnt)):
		buf[cnt[i]-1] = 1
		# if there is not valence array, return
		if cnt[i] == 0:
			return mean_vect

	# To reduce the complexity, if len(valence) has several values, return (must improve)
	num = 0
	for i in range(1,len(buf)):
		if buf[i] != 0:
			num = num+1
	if num > 1:
		return mean_vect

	y_now = np.dot(mean_vect,valences)
	y_min = np.min(y_now)

	# if mean_vect is already balance, return
	if iteration == 0:
		return mean_vect

	# set the covariance of gaussian (proposal distribution)
	cov = np.zeros((len(mean_vect),len(mean_vect)),dtype=np.float32)
	for i in range(len(mean_vect)):
		cov[i][i] = macro._SIGMA

	before_y_min = y_min
	before_mean_vect = mean_vect
	for ite in range(iteration):

		y_now = np.dot(mean_vect,valences)

		# if balance, return
		for i in range(len(y_now)):
			if abs(y_now[i]) < 1.0e-6:
				return mean_vect

		x_star = np.random.multivariate_normal(mean_vect,cov,macro._SAMPLING)
		x_star = x_star.clip(min=1.0e-6)
		for i in range(len(x_star)):
			x_star[i] = x_star[i]/np.sum(x_star[i])
		x_star = np.round(x_star,macro._ROUND)

		# expand array list
		bak = np.dot(x_star,valences)
		y_star = np.zeros((macro._SAMPLING,len(y_now)),dtype=np.float32)
		for i in range(len(y_star)):
			y_star[i] = bak[i]

		A = np.exp(-y_star*y_star)/np.exp(-y_now*y_now)
		max_A = np.max(A)
		arg_A_x, arg_A_y = np.unravel_index(np.argmax(A),A.shape)

		if max_A > 1.0:
			max_A = 1.0
		p = np.random.random()
		if p < max_A and np.sum(mean_vect) > 1.0e-6:
			for i in range(len(mean_vect)):
				mean_vect[i] = x_star[arg_A_x][i]
		if y_star[arg_A_x][arg_A_y] < 1.0e-6:
			break

	return mean_vect
