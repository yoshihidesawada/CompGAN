# GAN for generating chemical compositions
Sample code of CondGAN for Inorganic Chemical Compositions

# Dependencies
- Python 3.6.3::Anaconda Custom (64-bit)
- tensorflow 1.10.0
- pandas 0.20.3
- numpy 1.14.2
- xenonpy 0.3.2
- pymatgen 2018.4.6

# Usage
Before executing this code, please make ./inputs/training.csv, ./outputs, and ./tmp. Then, please execute following command.

`python main.py`

After finishing, models and generated compositions are saved in ./outputs. In ./tmp, atom list, normalization parameters, and training data with physic descriptors are saved.

# Citation
Yoshihide Sawada, Koji Morikawa, Mikiya Fujii, "Study of Deep Generative Models for Inorganic Chemical Compositions", 
https://arxiv.org/abs/1910.11499

# Copyright
Copyright (c) 2019 Yoshihide Sawada
Released under the MIT license
https://opensource.org/licenses/mit-license.php
