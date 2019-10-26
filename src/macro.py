# used in "main.py"
_SAVE_NORMALIZATION_PARAM='../tmp/normalization_param.csv' # save file for normalization parameters
_LAMB=10.0 # hyperparameter of W-GAN-GP
_MAX_EPOCH=50000 # maximum number of iteration
_MAX_EPOCH_TRAIN_DISCRIMINATOR=5 # inner loop that only trains discriminator
_FILEPATH='../outputs/' # directory for saving models

# used in "download_data.py" to save training dataset
_INORGANIC_DATA='../inputs/training.csv'
_LOCAL_SAVE_DATA='../tmp/training_with_desc.csv'
_ATOM_LIST='../tmp/atom_list.csv'

# mainly used in "preprocess.py" to normalize
_EPS=1.0e-6 # hyperparameter to avoid 0 division

# used in "model.py"
_BATCH_SIZE=256 # batch size
_NOISE_DIM=10 # dimension of noise to feed into generator
_LAYER_DIM=30 # hyperparameter of dimensions of each layer
_PROP_DIM=1 # dimension of target property

# used in "evaluation.py"
_SAVE_FILE='generated_compositions.csv'
_TEST_BATCH_SIZE=10 # the number of generated compositions
_TH=0.03 # if bag-of-atoms vector x is lower than _TH, then x=0
_TARGET_PROP=-3.0 # target property for generating compositions

# used in "mcmc.py"
_ITERATION=10000
_SAMPLING=256
_SIGMA=0.05
_ROUND=2 # rounding e.g, 1.234 -> 1.23
