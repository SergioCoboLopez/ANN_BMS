import pyrenn
import numpy as np
import copy
import pandas as pd





function='leaky_ReLU'
n=6
sigma=0.0
realization=2
resolution=0.01

#Read high res data
#----------------------------------
file_name_d='NN_function_tanh_NREP_10_res_' + str(resolution) + '_data.csv'
file_name_d='../data/generative_data/' + file_name_d
d=pd.read_csv(file_name_d)
d=d.drop(columns='Unnamed: 0')
dn=d[d['rep']==n]
dn=dn.reset_index(drop=True)
#----------------------------------

#Read neural network
#----------------------------------
file_name_nn='NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) +\
    '_rep_' + str(n) + '_r_' + str(realization) + '.csv'

file_name_nn='../data/trained_nns/' + file_name_nn

nn=pyrenn.loadNN(file_name_nn)
#----------------------------------

#predictions of nn
train_value=1.0
train_border_row=dn[(dn['x1']<=1.0) & (dn['x1']>=0.99)]
train_size=train_border_row.index[0]

print(train_border_row.index[0])

x_tot=dn['x1']

ytest=pyrenn.NNOut(x_tot, nn)

print(ytest)





