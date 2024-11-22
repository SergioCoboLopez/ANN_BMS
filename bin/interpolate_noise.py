#19/11/2024. This code takes datasets with double resolution as the original datasets. Then takes the noise signal from the original datasets and interpolates the noise, by taking the mean of the noise between any two points.

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

function='tanh' #tanh, leaky_ReLU
mean=0;sigma=0.2;realization=0
sigmas=[i for i in np.arange(0,0.22,0.02)]

#Read double resolution file
file_data='NN_function_' + function + '_NREP_10_res_0.025_data' + '.csv'
data='../data/generative_data/' + file_data
d=pd.read_csv(data)
d=d.drop(columns='Unnamed: 0')
print(d)

for sigma in sigmas:
    for r in range(3):
        
        print(sigma, realization)
            
        #Read noisy signal files
        if sigma==0.1:
            file_noise='NN_' + function + '_sigma_' + '0.10' + '_r_' + str(r) + '.csv'
        else:
            file_noise='NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(r) + '.csv'
        
        noise='../data/' + file_noise
        n=pd.read_csv(noise)
        n=n.drop(columns='Unnamed: 0')
        print(n)

        #noise column to list
        noise=n['noise'].tolist()

        #Add noise to high resolution data
        high_res_noise=[]
        np.random.seed(seed=1111)
        end_point_noise = np.random.normal(mean,sigma,1)
        for i in range(len(noise)):
            high_res_noise.append(noise[i]) #copy noise for existing point
            try:
                high_res_noise.append(0.5*(noise[i] + noise[i+1])) #interpolate noise for new points

            except IndexError:
                high_res_noise.append(end_point_noise[0]) #Add new noise for final point

        d['noise']=high_res_noise
        d['y_noise']= d['y'] + d['noise']
        
        #Save data
        d.to_csv('../data/2x_resolution/' + 'NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(r) + '_res_0.025' +  '.csv')

        


#Read noisy signal files
# file_noise='NN_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '.csv'
# noise='../data/' + file_noise
# n=pd.read_csv(noise)
# n=n.drop(columns='Unnamed: 0')
# print(n)

# #noise column to list
# noise=n['noise'].tolist()

# high_res_noise=[]

# mean=0
# point_noise = np.random.normal(mean,sigma,1)
# print(point_noise)

# for i in range(len(noise)):
#     high_res_noise.append(noise[i])
#     try:
#         high_res_noise.append(0.5*(noise[i] + noise[i+1]) )

#     except IndexError:
#         high_res_noise.append(point_noise[0])

# print(len(noise))

# print(len(high_res_noise))

# d['noise']=high_res_noise
# d['ynoise']= d['y'] + d['noise']


# # function_' + function + '_NREP_10_data' + '.csv'
# # model_d='../data/' + file_model
# # d=pd.read_csv(model_d)
# # d=d.drop(columns='Unnamed: 0')
