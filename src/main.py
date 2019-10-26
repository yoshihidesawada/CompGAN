import os
import sys

import numpy as np
import pandas as pd

from functools import partial

# tf.keras definition
# these may change depending on your install environments
from tensorflow.contrib.keras.api.keras import backend, callbacks
from tensorflow.contrib.keras.api.keras.models import Model, load_model
from tensorflow.contrib.keras.api.keras.layers import Input
from tensorflow.contrib.keras.api.keras.utils import Progbar
from tensorflow.contrib.keras.api.keras.optimizers import Adam

import download_data as dl # download_data.py
import preprocess as pre # preprocess.py
import evaluation as eval # evaluation.py
import model # model.py
import macro # macro.py


def main():

	if os.path.isfile(macro._LOCAL_SAVE_DATA)==0:

		# download data and compute featuers (see "download_data.py")
		# atomic_numbers use to compute composition vector
		# labels is target properties (formation energy)
		train_labels, compositions, features, atomic_numbers = dl.get_data()

		# compute bag-of-atom vector that trains GAN (see "preprocess.py")
		boa_vectors = pre.compute_bag_of_atom_vector(compositions, atomic_numbers)
		train_data = np.concatenate([boa_vectors,features],axis=1)

		save_data = pd.DataFrame(np.concatenate([train_labels,train_data],axis=1))
		save_data.to_csv(macro._LOCAL_SAVE_DATA, index=False, header=False)

	else:
		data = pd.read_csv(macro._LOCAL_SAVE_DATA,delimiter=',',engine="python",header=None)
		data = np.array(data)
		train_labels, train_data = np.split(data,[1],axis=1)

	# normalization of training data such that min is 0 and max is 1 (see "preprocess.py")
	normalized_train_data, data_max, data_min = pre.normalize_for_train(train_data)
	normalized_train_labels, max_train_prop, min_train_prop = pre.normalize_for_train(train_labels)

	# Save normalization parameter to .csv to use generation
	save_data = pd.DataFrame(np.concatenate([max_train_prop,min_train_prop,data_max,data_min],axis=0))
	save_data.to_csv(macro._SAVE_NORMALIZATION_PARAM, index=False, header=False)

	### start initialization of training GAN ###

	# set hyperparameters
	batch_size = macro._BATCH_SIZE # batch size
	noise_dim = macro._NOISE_DIM # dimension of noise to input generator
	property_dim = macro._PROP_DIM # the number of properties
	lamb = macro._LAMB # hyperparameter for W-GAN-GP
	max_epoch = macro._MAX_EPOCH # maximum iteration of outer loop
	max_train_only_dis = macro._MAX_EPOCH_TRAIN_DISCRIMINATOR # maximum iteration of inner loop defined by W-GAN-GP paper (https://arxiv.org/pdf/1704.00028.pdf)
	max_loop = int(train_data.shape[0]/batch_size)

	# set model (see "model.py")
	# in this code, we apply AC-GAN based network architecture (https://arxiv.org/abs/1610.09585)
	# difference between AC-GAN is that our model is the regression, not classification
	gen = model.generator(normalized_train_data.shape[1])
	dis = model.discriminator(normalized_train_data.shape[1])

	# rf is the output layer of discriminator that discriminates real or fake
	rf = model.real_fake()

	# pred is the output layer of discriminator that predicts target property
	pred = model.prediction()

	# set optimization method
	dis_opt = Adam(lr=1.0e-4, beta_1=0.0, beta_2=0.9)
	gen_opt = Adam(lr=1.0e-4, beta_1=0.0, beta_2=0.9)

	# first set discriminator's parameters for training
	gen.trainable = False # generator's parameter does not update
	dis.trainable = True
	rf.trainable = True
	pred.trainable = True

	# set variables when inputting real data
	real_inputs = Input(shape=normalized_train_data.shape[1:])
	dis_real_outputs = dis(real_inputs)
	real_fake_from_real = rf(dis_real_outputs)
	predictions_from_real = pred(dis_real_outputs)

	# set variables when inputting fake data
	fake_inputs = Input(shape=(noise_dim+property_dim,))
	gen_fake_outputs = gen(fake_inputs)
	dis_fake_outputs = dis(gen_fake_outputs)
	real_fake_from_fake = rf(dis_fake_outputs)

	# set loss function for discriminator
	# in this case, we apply W-GAN-GP based loss function because of improving stability
	# W-GAN-GP (https://arxiv.org/pdf/1704.00028.pdf)
	# W-GAN-GP is unsupervised training, on the other hand, our model is supervised (conditional).
	# So, we apply wasserstein_loss to real_fake part and apply mean_squared_error to prediction part
	interpolate = model.RandomWeightedAverage()([real_inputs,gen_fake_outputs])
	dis_interpolate_outputs = dis(interpolate)
	real_fake_interpolate = rf(dis_interpolate_outputs)

	# gradient penalty of W-GAN-GP
	gp_reg = partial(model.gradient_penalty,interpolate=interpolate,lamb=lamb)
	gp_reg.__name__ = 'gradient_penalty'

	# connect inputs and outputs of the discriminator
	# prediction part is trained by only using training dataset (i.e., predict part is not trained by generated samples)
	dis_model = Model(inputs=[real_inputs, fake_inputs],\
	outputs=[real_fake_from_real, real_fake_from_fake, real_fake_interpolate, predictions_from_real])

	# compile
	dis_model.compile(loss=[model.wasserstein_loss,model.wasserstein_loss,\
	gp_reg,'mean_squared_error'],optimizer=dis_opt)


	# second set generator's parameters for training
	gen.trainable = True # generator's parameters only update
	dis.trainable = False
	rf.trainable = False
	pred.trainable = False

	# set variables when inputting noise and target property
	gen_inputs = Input(shape=(noise_dim+property_dim,))
	gen_outputs = gen(gen_inputs)

	# set variables for discriminator when inputting fake data
	dis_outputs = dis(gen_outputs)
	real_fake = rf(dis_outputs)
	predictions = pred(dis_outputs)

	# connect inputs and outputs of the discriminator
	gen_model = Model(inputs=[gen_inputs],outputs=[real_fake,predictions])

	# compile
	# generator is trained by real_fake classification and prediction of target property
	gen_model.compile(loss=[model.wasserstein_loss,'mean_squared_error'], optimizer=gen_opt)


	# if you need progress bar
	progbar = Progbar(target=max_epoch)

	# set the answer to train each model
	real_label = [-1]*batch_size
	fake_label = [1]*batch_size
	dummy_label = [0]*batch_size

	#real = np.zeros((batch_size,train_data.shape[1]), dtype=np.float32)
	inputs = np.zeros((batch_size,noise_dim+property_dim), dtype=np.float32)

	# epoch
	for epoch in range(max_epoch):

		# iteration
		for loop in range(max_loop):

			# shuffle to change the trainng order and select data
			sdata, slabels, bak = pre.paired_shuffle(normalized_train_data,normalized_train_labels)
			real = sdata[loop*batch_size:(loop+1)*batch_size]
			properties = slabels[loop*batch_size:(loop+1)*batch_size]

			# generator's parameters does not update
			gen.trainable = False
			dis.trainable = True
			rf.trainable = True
			pred.trainable = True

			# train discriminator
			for train_only_dis in range(max_train_only_dis):
				noise = np.random.uniform(-1,1,(batch_size,noise_dim)).astype(np.float32)
				for i in range(len(noise)):
					inputs[i] = np.hstack((noise[i],properties[i]))
				dis_loss = dis_model.train_on_batch([real,inputs],[real_label,fake_label,dummy_label,properties])

			# second train only generator
			gen.trainable = True
			dis.trainable = False
			rf.trainable = False
			pred.trainable = False
			noise = np.random.uniform(-1,1,(batch_size,noise_dim)).astype(np.float32)
			for i in range(len(noise)):
				inputs[i] = np.hstack((noise[i],properties[i]))
			gen_loss = gen_model.train_on_batch([inputs],[real_label,properties])

		# if you need progress bar
		progbar.add(1, values=[("dis_loss",dis_loss[0]), ("gen_loss", gen_loss[0])])


	# save generated samples and models
	eval.save(normalized_train_data, gen, dis, pred, rf)

	backend.clear_session()


if __name__ == '__main__':
	main()
