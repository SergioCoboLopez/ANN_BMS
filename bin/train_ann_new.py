# Train neural networks with pyrenn to predict ANN-generated functions without overfitting.
# In this code, I split the training set in training (50 points) and validation (10 points).
# I do two iterations
import pyrenn
import numpy as np
import copy
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error


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

#print(d)
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)
print(d[d['rep']==0])
#-----------------------------------------------------

#Build ANN
ILS = 1;OLS=1
NL, LS = 5, 10
arch=[ILS] + NL*[LS] + [OLS]
nn=pyrenn.CreateNN(arch)

#Train
train_size=50
validation_size=train_size + 10
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

    #Build validation set
    xvalid=dn.loc[train_size:validation_size-1]['x1']
    yvalid=dn.loc[train_size:validation_size-1]['y_noise']

    #Error and neural network vectors
    MAE=[];MSE=[];RMSE=[]        #Lists of validation errors
    MAE_t=[];MSE_t=[]; RMSE_t=[] #List of training errors
    nn_dict={} #Dictionary of neural network models

    for i in range(30):
        #Two iterations of training a neural network (k_max=1)
        net=pyrenn.train_LM(xtrain,ytrain,nn,verbose=True,k_max=1,E_stop=1e-200)
        
        #Test NN on validation set
        yvalid_test = pyrenn.NNOut(xtrain,net) #Prediction on training test
        yvalid_pred = pyrenn.NNOut(xvalid,net) #Prediction on validation set

        #Validation errors
        #--------------------------------------------------
        MSE_i=mean_squared_error(yvalid,yvalid_pred)
        MSE.append(MSE_i)

        RMSE_i=root_mean_squared_error(yvalid,yvalid_pred)
        RMSE.append(RMSE_i)
        #--------------------------------------------------

        #Training errors
        #--------------------------------------------------
        MSE_t_i=mean_squared_error(ytrain,yvalid_test)
        MSE_t.append(MSE_t_i)

        RMSE_t_i=root_mean_squared_error(ytrain,yvalid_test)
        RMSE_t.append(RMSE_t_i)
        #--------------------------------------------------

        #deepcopy and save neural network to dictionary
        net_copy=copy.deepcopy(net)
        nn_dict[i]=net_copy
        
        #update neural network for next step of the loop
        nn=net
        
        # print(nn_dict[0]['w'][0])
        # print(nn_dict[i]['w'][0])
        # print("\n")

        
    #Find the model with the minimum error
    min_error_mse=min(MSE);min_error_rmse=min(RMSE)
    print(min_error_mse,min_error_rmse)

    #Take indices of the elements with minimum error
    min_err_mse_ind=MSE.index(min_error_mse);min_err_rmse_ind=RMSE.index(min_error_rmse)
    print(min_err_mse_ind, min_err_rmse_ind)

    #--------------------------------------------------------
#    plt.plot(MAE, '.', color='blue', label='MAE validation')
    plt.plot(MSE, '.', color='red',label='MSE validation')
    plt.plot(RMSE,'.', color='green',label='RMSE validation')

#    plt.plot(MAE_t, linewidth=1,linestyle='--',color='blue',label='MAE train')
    plt.plot(MSE_t, linewidth=1,linestyle='--',color='red',label='MSE train')
    plt.plot(RMSE_t,linewidth=1,linestyle='--',color='green',label='RMSE train')
    plt.legend(loc='best')
    #--------------------------------------------------------
    plt.show()

    
    #Save neural network
    if noise==True:
            pyrenn.saveNN(net,'../data/'+ 'NN_weights_TEST_noise_' + activation_function + '_train_' + str(train_size) + '_rep_' + str(n) + '.csv')
    else:
            pyrenn.saveNN(net,'../data/'+ 'NN_weights_TEST_' + activation_function + '_train_' + str(train_size) + '_rep_' + str(n) + '.csv')
    
    #First NN found
    net_first=nn_dict[0]
    xtest = dn.loc[train_size:]['x1']
    ytest_first = pyrenn.NNOut(xtrain,net_first)
    ypred_first = pyrenn.NNOut(xtest,net_first)

    #save results
    ymodel_n=np.concatenate((ytest_first, ypred_first))

    #Last NN found
    net_last=nn_dict[9]
    print(net_last['w'][0])
    xtest = dn.loc[train_size:]['x1']
    ytest_last = pyrenn.NNOut(xtrain,net_last)
    ypred_last = pyrenn.NNOut(xtest,net_last)

    #save results
    ymodel_last=np.concatenate((ytest_last, ypred_last))

    #Best nn found
    #------------------------------------------------------
    net_best=nn_dict[min_err_mse_ind]
    print(net_best['w'][0])
    ytest_best = pyrenn.NNOut(xtrain,net_best)
    ypred_best = pyrenn.NNOut(xtest,net_best)
    ymodel_best=np.concatenate((ytest_best, ypred_best))
    #------------------------------------------------------

    #Save neural network
    if noise==True:
            pyrenn.saveNN(net_best,'../data/'+ 'NN_weights_no_overfit_noise_' + activation_function + '_train_' + str(train_size) + '_rep_' + str(n) + '.csv')
    else:
            pyrenn.saveNN(net_best,'../data/'+ 'NN_weights_no_overfit_' + activation_function + '_train_' + str(train_size) + '_rep_' + str(n) + '.csv')
    
    xplot=np.concatenate((xtrain,xtest))
    plt.plot(xplot,ymodel_n,'k', label='first model')
    plt.plot(xplot,ymodel_last,'r', label='last model')
    plt.plot(xplot,ymodel_best,'green',label='best model') 
    plt.plot(xplot,dn.y_noise, 'orange', label='noise')
    plt.plot(xplot,dn.y, 'blue', label='original')
    plt.legend(loc='best')
    plt.show()

    try:
        ymodel=np.append(ymodel,ymodel_n)
    except NameError:
        ymodel=ymodel_n


#Add predictions and save data
d['ymodel']=ymodel
if noise==True:
    d.to_csv('../data/'+ 'NN_model_best_noise_no_overfit_' + activation_function +  '_train_' + str(train_size) + '_NREP_10_data' + '.csv')
else:
    d.to_csv('../data/'+ 'NN_model_no_overfit_' + activation_function +  '_train_' + str(train_size) + \
         '_NREP_10_data' + '.csv')





