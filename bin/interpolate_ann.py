import pyrenn
import numpy as np
import copy
import pandas as pd
import sys


#CAUTION!! You need to activate a virtual environment to run this code. In your terminal type:
#source ~/entorno/bin/activate. If you don't have a virtual environment in your computer, create
#one with numpy 1.20 and pandas 1.2.5 so that pyrenn can work without conflicting with numpy
#To deactivate the virtual environment type "deactivate"


function='leaky_ReLU' #tanh, leaky_ReLU

realization=sys.argv[1]
sigma=sys.argv[2]

resolution=0.01

#Read high res data
#----------------------------------
file_name_d='NN_function_' + function + '_NREP_10_res_' + str(resolution) + '_data.csv'
file_name_d='../data/generative_data/' + file_name_d
d=pd.read_csv(file_name_d)
d=d.drop(columns='Unnamed: 0')
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)

n_functions=int(d['rep'].max()) #Number of functions in dataset

for n in range(n_functions + 1):

    dn=d[d['rep']==n]
    dn=dn.reset_index(drop=True)
    #----------------------------------

    #Read neural network
    #----------------------------------
    file_name_nn='NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) +\
        '_rep_' + str(n) + '_r_' + str(realization) + '.csv'

    file_name_nn='../data/1x_resolution/trained_nns/' + file_name_nn

    nn=pyrenn.loadNN(file_name_nn)
    #----------------------------------

    #predictions of nn

    x_tot=dn['x1']
    
    #Save results
    #----------------------------------
    ymodel=pyrenn.NNOut(x_tot, nn)
    try:
        ymodels_all=np.append(ymodels_all,ymodel)
    except NameError:
        ymodels_all=ymodel
    #----------------------------------

#Save updated data with model
d['ymodel']=ymodels_all
d.to_csv('../data/inter_extrapolate_nns/'+ 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '_res_0.01.csv')
