#24/1/25. Following up on similar codes, this one adds noise to the nguyen datasets

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

np.random.seed(seed=1111)

mean=0;sigma=0.2;realization=0
sigmas=[i for i in np.arange(0,0.22,0.02)]

#Read nguyen data
file_data='nguyen_data_all.csv'
data='../../data/generative_data/' + file_data
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
        d['z_noise']= d['z'] + d['noise']

        #Save data                                                                                  
        d.to_csv('../../data/nguyen/1x_resolution/' + 'NN_nguyen'+ '_sigma_' + str(sigma) + '_r_' + str(r) +  '.csv')
