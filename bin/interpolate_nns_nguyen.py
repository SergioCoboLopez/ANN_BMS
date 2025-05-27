import pyrenn
import numpy as np
import copy
import pandas as pd
import sys


#CAUTION!! You need to activate a virtual environment to run this code. In your terminal type:
#source ~/entorno/bin/activate. If you don't have a virtual environment in your computer, create
#one with numpy 1.20 and pandas 1.2.5 so that pyrenn can work without conflicting with numpy
#To deactivate the virtual environment type "deactivate"

resolution='1x' #1x, 2x, 0.5x, 4e-3x
resolutions={'0.5x':'', '1x':'0.01', '2x': '', '4e-3x':'' }

function=sys.argv[1] #1, 5, 7, 8, 10
sigma=sys.argv[2]
realization=sys.argv[3]

#Read high resolution data to interpolate
#----------------------------------
input_path= '../data/generative_data/'
file_name_d=input_path + 'NN_function_'+function + '_NREP_10_res_' + str(resolutions[resolution]) + '_interpolate_data.csv'

output_path='../data/inter_extrapolate_nns/'


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
    nn_path='../data/' + str(resolution) + '_resolution/trained_nns/'
    file_name_nn=nn_path + 'NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) +\
        '_rep_' + str(n) + '_r_' + str(realization) + '.csv'

    nn=pyrenn.loadNN(file_name_nn)
    #----------------------------------

    #predictions of nn
    x_tot=dn['x1']
    ymodel=pyrenn.NNOut(x_tot, nn)
    
    #Save results
    #----------------------------------
    try:
        ymodels_all=np.append(ymodels_all,ymodel)
    except NameError:
        ymodels_all=ymodel
    #----------------------------------

#Save updated data with model
d['ymodel']=ymodels_all
d.to_csv( output_path + 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '_res_'  + resolutions[resolution] + '.csv')
# d.to_csv('../data/inter_extrapolate_nns/'+ 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '_res_0.01.csv')
