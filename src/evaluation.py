import os
import sys
import numpy as np
import pandas as pd
from pymatgen import Element, Composition

import macro
import mcmc

# save resutls to evaluate
def save(normalized_data, gen, dis, pred, rf):

    # load parameters
    atom_list = np.array(pd.read_csv(macro._ATOM_LIST, delimiter=',',engine="python",header=None))
    data = np.array(pd.read_csv(macro._SAVE_NORMALIZATION_PARAM,delimiter=',',engine="python",header=None))
    max_train_prop, min_train_prop, data_max, data_min = np.split(data,[1,2,normalized_data.shape[1]+2], axis=0)

    batch_size = macro._TEST_BATCH_SIZE # the number of generated samples of each input target property
    noise_dim = macro._NOISE_DIM # nosie dimenstion
    property_dim = macro._PROP_DIM # target property dimension
    property = (macro._TARGET_PROP-min_train_prop)/(max_train_prop-min_train_prop+macro._EPS) # normalized target property

    # evaluation
    # feed all target properties in test data into generator
    # compute nearest neighbor method between generate sample and test data
    # save generated composition, corresponding nearest composition, and corresponding target property
    # generate samples based on the test target property
    noise = np.random.uniform(-1,1,(batch_size,noise_dim)).astype(np.float32)
    inputs = np.zeros((batch_size,noise_dim+property_dim), dtype=np.float32)
    for i in range(len(noise)):
        inputs[i] = np.hstack((noise[i],property[0][0]))
    fake = np.array(gen.predict(inputs))
    fake_last_layer = dis.predict(fake)

    # renormalized generated samples
    renormalized_fake = fake*(data_max.T-data_min.T+macro._EPS) + data_min.T

    # thresholding composition vector
    # if value of composition vector is lower than macro._TH, set 0 (to guarantee sparsity)
    boa_vectors = []
    for i in range(len(noise)):
        boa_vector = renormalized_fake[i,:len(atom_list)]
        for j in range(len(boa_vector)):
            if abs(boa_vector[j]) < macro._TH:
                boa_vector[j] = 0.0
        if np.sum(boa_vector) != 0.0:
            boa_vector = boa_vector/np.sum(boa_vector)
        boa_vectors.append(boa_vector)
    boa_vectors = np.array(boa_vectors)

    # convert composition vector to chemical composition
    generated_compositions = []
    for i in range(len(boa_vectors)):

        valences = []
        for j in range(len(boa_vectors[0])):
            if np.abs(boa_vectors[i][j]) > 1.0e-6:
                valence = []
                for k in range(len(Element(atom_list[j][0]).common_oxidation_states)):
                    valence.append(float(Element(atom_list[j][0]).common_oxidation_states[k]))
                valences.append(np.array(valence))
        valences = np.array(valences)
        mcmc_vectors = mcmc.metropolis_hastings(valences,boa_vectors[i],macro._ITERATION)

        generated_composition = ''
        for j in range(len(mcmc_vectors)):
            if np.abs(mcmc_vectors[j]) > 0.0:
                generated_composition = generated_composition+atom_list[j]+"%.2f"%(mcmc_vectors[j])

        generated_compositions.append(generated_composition)

    # save generated result
    fp = open('%s/%s'%(macro._FILEPATH,macro._SAVE_FILE),'w')
    for i in range(len(generated_compositions)):
        fp.write('%s\n'%(generated_compositions[i][0]))
    fp.close()

    # save models
    gen.save('%s/generator.h5'%(macro._FILEPATH))
    dis.save('%s/discriminator.h5'%(macro._FILEPATH))
    rf.save('%s/real_fake_classifier.h5'%(macro._FILEPATH))
    pred.save('%s/prediction.h5'%(macro._FILEPATH))
