#19/11/2024. This code takes datasets with double resolution as the original datasets. Then takes the noise signal from the original datasets and interpolates the noise, by taking the mean of the noise between any two points.

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

np.random.seed(seed=1111)

function='leaky_ReLU' #tanh, leaky_ReLU
mean=0;sigma=0.2;realization=0
sigmas=[i for i in np.arange(0,0.22,0.02)]

#Read double resolution file
file_data='NN_function_' + function + '_NREP_10_res_0.1_data' + '.csv'
data='../data/generative_data/' + file_data
d=pd.read_csv(data)
d=d.drop(columns='Unnamed: 0')
print(d)

sample=d.index.stop
print(d.index.stop)


for sigma in sigmas:
    for r in range(3):
        
        print(sigma, r)
        noise = np.random.normal(mean,sigma,sample)            

        #Add noise to high resolution data
        d['noise']=noise
        d['y_noise']= d['y'] + d['noise']
        
        #Save data
        d.to_csv('../data/0.5x_resolution/' + 'NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(r) + '_res_0.1' +  '.csv')
