# Train neural networks with pyrenn to predict ANN-generated functions
import pyrenn
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec

#Read data
#-----------------------------------------------------
noise=True
activation_function='leaky_ReLU'


if noise==True:
    filename='NN_noisy_signal_' + activation_function + '.csv'
else:
    filename='NN_function_' + activation_function + '_NREP_10_data.csv'

data='../data/' + filename

d=pd.read_csv(data)
d=d.drop(columns='Unnamed: 0')

print(d)
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)
#-----------------------------------------------------

#Build ANN
ILS = 1;OLS=1
NL, LS = 5, 10
arch=[ILS] + NL*[LS] + [OLS]
nn=pyrenn.CreateNN(arch)

#Empty dataframe for NN weights
#d_weights = pd.DataFrame()

#Train
train_size=60
n_functions=int(d['rep'].max())
print(n_functions)

for n in range(n_functions + 1):    
    #Read data
    dn=d[d['rep']==n]
    dn.index.name = None
    dn=dn.reset_index(drop=True)
    
    #Train NN
    #Train on the  first points
    xtrain = dn.loc[0:train_size-1]['x1']
    ytrain = dn.loc[0:train_size-1]['y_noise']


    net=pyrenn.train_LM(xtrain,ytrain,nn,verbose=True,k_max=100,E_stop=1e-5)

    
    #Save neural network
    if noise==True:
            pyrenn.saveNN(net,'../data/'+ 'NN_weights_noise_' + activation_function + '_train_' + str(train_size) + '_rep_' + str(n) + '.csv')

    else:
            pyrenn.saveNN(net,'../data/'+ 'NN_weights_' + activation_function + '_train_' + str(train_size) + '_rep_' + str(n) + '.csv')
    
    #Test NN
    xtest = dn.loc[train_size:]['x1']
    ytest = pyrenn.NNOut(xtrain,net)
    ypred = pyrenn.NNOut(xtest,net)

    #save results
    ymodel_n=np.concatenate((ytest, ypred))
    try:
        ymodel=np.append(ymodel,ymodel_n)
    except NameError:
        ymodel=ymodel_n


#Add predictions and save data
d['ymodel']=ymodel
if noise==True:
    d.to_csv('../data/'+ 'NN_model_noise_' + activation_function +  '_train_' + str(train_size) + '_NREP_10_data' + '.csv')
else:
    d.to_csv('../data/'+ 'NN_model_' + activation_function +  '_train_' + str(train_size) + \
         '_NREP_10_data' + '.csv')





