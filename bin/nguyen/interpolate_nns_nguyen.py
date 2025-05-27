import pyrenn
import numpy as np
import copy
import pandas as pd
import sys


#CAUTION!! You need to activate a virtual environment to run this code. In your terminal type:
#source ~/entorno/bin/activate. If you don't have a virtual environment in your computer, create
#one with numpy version 1.20 and pandas version 1.2.5 so that pyrenn can work without conflicting with numpy.
#To deactivate the virtual environment type "deactivate"


#function='leaky_ReLU' #tanh, leaky_ReLU

realization=sys.argv[1]
sigma=sys.argv[2]

#Read high res data
#----------------------------------
input_path='../../data/generative_data/'
file_name='nguyen_inter_extrapolation_data_all.csv'
d=pd.read_csv(input_path + file_name)
d=d.drop(columns='Unnamed: 0')
d=d.reset_index(drop=True)
print(d)

id_functions=[1,5,7,8,10]

print(d)

for n in id_functions:
    print(n)

    dn=d[d['rep']==n]
    dn=dn.reset_index(drop=True)
    #----------------------------------

    #Read neural network
    #----------------------------------
    nn_path='../../data/nguyen/1x_resolution/trained_nns/' 
    file_name_nn='NN_weights_no_overfit_sigma_' + str(sigma)+ '_rep_' + str(n) + \
                 '_r_' + str(realization) + '.csv'

    nn=pyrenn.loadNN(nn_path + file_name_nn)
    #----------------------------------

    #predictions of nn
    x_tot=dn['x']
    
    #Save results
    #----------------------------------
    ymodel=pyrenn.NNOut(x_tot, nn)
    try:
        ymodels_all=np.append(ymodels_all,ymodel)
    except NameError:
        ymodels_all=ymodel
    #----------------------------------

#Save updated data with model
d['zmodel']=ymodels_all
d.to_csv('../../data/nguyen/inter_extrapolation/' + 'NN_no_overfit_inter_extrapolation_sigma_' +\
         str(sigma) + '_r_' + str(realization) + '.csv')
