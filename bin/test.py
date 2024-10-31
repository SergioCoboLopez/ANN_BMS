import pyrenn
import numpy as np
import pandas as pd

net = pyrenn.loadNN('../data/trained_nns/NN_weights_no_overfit_tanh_sigma_0.2_rep_6_r_1.csv')

print(net)
