#10/9/2024
# This code trains neural networks in input data without overfitting.
# The train set is split in another training set (50 points) and a validation set (10 points). The number of points may change in future versions of this code.
# The training is done in sets of two epochs and there are n iterations of these sets. After each set, the errors (MAE and RMSE) are evaluated.
#The code saves the best neural network. Best, being the neural network with the minimum RMSE on the validation set, meaining that there is no overfitting.
#The code plots the errors (RMSE and MAE) as a function of the iterations. It also plots the first nn, the last nn, and the best nn together with the original signal and the signal with noise.
#In this version, MAE is purely informative and the best model is based on the minimum RMSE.

import pyrenn
import numpy as np
import copy
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import sys
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error


#Read data
#-----------------------------------------------------
function='tanh'
sigma=0.14

#realization=0
#sigma=int(sys.argv[1])
realization=int(sys.argv[1])

resolution='2x'


if resolution=='1x':
    filename='NN_' + function + '_sigma_' + str(sigma) + '_r_' +str(realization) + '.csv'
    if sigma==0.10:
        filename='NN_' + function + '_sigma_' + '0.1' + '_r_' +str(realization) + '.csv'
        
    data='../data/1x_resolution/' + filename
    train_size=50;validation_size=train_size + 10
    output_path='../data/1x_resolution/trained_nns/'
    
elif resolution=='2x':
    filename='NN_' + function + '_sigma_' + str(sigma) + '_r_' +str(realization) + '_res_0.025.csv'
    # if sigma==0.10:
    #      filename='NN_' + function + '_sigma_' + '0.10' + '_r_'+str(realization)+'_res_0.025.csv'
    
    data='../data/2x_resolution/' + filename
    train_size=100;validation_size=train_size + 20
    output_path='../data/2x_resolution/trained_nns/'
    
elif resolution=='0.5x':
    filename='NN_' + function + '_sigma_' + str(sigma) + '_r_' +str(realization) + '_res_0.1.csv'
    data='../data/0.5x_resolution/trained_nns/' + filename

    # if sigma==0.10:
    #     filename='NN_' + function + '_sigma_' + '0.10' + '_r_' +str(realization) + '_res_0.1.csv'

    data='../data/0.5x_resolution/' + filename
    train_size=25;validation_size=train_size + 5
    output_path='../data/0.5x_resolution/trained_nns/'


d=pd.read_csv(data)

d=d.drop(columns='Unnamed: 0')
#Take subset of data
d=d[(d['x1'] >= -2.0) & (d['x1']<=2.0)]
d=d.reset_index(drop=True)
#-----------------------------------------------------

#Build ANN
ILS = 1;OLS=1
NL, LS = 5, 10
arch=[ILS] + NL*[LS] + [OLS]
nn=pyrenn.CreateNN(arch)

#Cross validations
train_border=d[d['rep']==0].loc[train_size-1]['x1']
valid_border=d[d['rep']==0].loc[validation_size-1]['x1']
n_functions=int(d['rep'].max()) #Number of functions in dataset
iterations=300


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

#--------------------------------------------------
    for i in range(iterations):
        #Two iterations of training a neural network (k_max=1)
        net=pyrenn.train_LM(xtrain,ytrain,nn,verbose=True,k_max=1,E_stop=1e-200)
        
        #Test NN on validation set
        yvalid_test = pyrenn.NNOut(xtrain,net) #Prediction on train
        yvalid_pred = pyrenn.NNOut(xvalid,net) #Prediction on valid

        #Validation errors
        #--------------------------------------------------
        MSE_i=mean_squared_error(yvalid,yvalid_pred)
        MSE.append(MSE_i)

        MAE_i=mean_absolute_error(yvalid,yvalid_pred)
        MAE.append(MAE_i)

        RMSE_i=root_mean_squared_error(yvalid,yvalid_pred)
        RMSE.append(RMSE_i)
        #--------------------------------------------------

        #Training errors
        #--------------------------------------------------
        MSE_t_i=mean_squared_error(ytrain,yvalid_test)
        MSE_t.append(MSE_t_i)

        MAE_t_i=mean_absolute_error(ytrain,yvalid_test)
        MAE_t.append(MAE_t_i)
        
        RMSE_t_i=root_mean_squared_error(ytrain,yvalid_test)
        RMSE_t.append(RMSE_t_i)
        #--------------------------------------------------

        #deepcopy and save neural network to dictionary
        net_copy=copy.deepcopy(net)
        nn_dict[i]=net_copy
        
        #update neural network for next step of the loop
        nn=net
#--------------------------------------------------
        
    #Find the model with the minimum error
    min_error_mse=min(MSE);min_error_rmse=min(RMSE)
    print(min_error_mse,min_error_rmse)

    #Take indices of the elements with minimum error
    min_err_mse_ind=MSE.index(min_error_mse);min_err_rmse_ind=RMSE.index(min_error_rmse)
    #--------------------------------------------------------

    #Plot errors
    #----------------------------------------------------

    #Figure settings
    #--------------------------------
    output_path_fig='../results/nn_w_validation/'

    
    name_fig='validation_errors_' + 'sigma_' + str(sigma) + '_' + str(function) + '_' + str(n) + '_r_' + str(realization)
    extensions=['.png']   #Extensions to save figure   
    
    #Define figure size
    cm = 1/2.54 #convert inch to cm                                  
    width = 10*cm; height=8*cm
    fig=figure(figsize=(width,height), dpi=300)

    #Fonts and sizes
    size_axis=7;size_ticks=6;size_title=5
    line_w=1;marker_s=3
    #--------------------------------
    plt.plot(MAE, '.', markersize=6, color='blue', label='MAE validation')
    plt.plot(RMSE,'.',markersize=6,color='green',label='RMSE validation')

    plt.plot(MAE_t, linewidth=1,linestyle='--',color='blue',label='MAE train')
    plt.plot(RMSE_t,linewidth=1,linestyle='--',color='green',label='RMSE train')
    plt.scatter(min_err_rmse_ind,min_error_rmse,s=80,marker='*',color='red',label='minimum rmse')
    #--------------------------------------------------------

    #Labels
    plt.legend(loc='best', fontsize=size_ticks)
    plt.xlabel('iterations',fontsize=size_axis);plt.ylabel('error',fontsize=size_axis)
    plt.title('%s, n=%d' % (function, n),fontsize=size_title)
    ext=extensions[0]
    plt.savefig(output_path_fig+name_fig+ext,dpi=300)
    
    #First NN found
    net_first=nn_dict[0]
    xtest = dn.loc[train_size:]['x1']
    ytest_first = pyrenn.NNOut(xtrain,net_first)
    ypred_first = pyrenn.NNOut(xtest,net_first)
    #save results
    ymodel_n=np.concatenate((ytest_first, ypred_first))

    #Last NN found
    net_last=nn_dict[iterations-1]
    xtest = dn.loc[train_size:]['x1']
    ytest_last = pyrenn.NNOut(xtrain,net_last)
    ypred_last = pyrenn.NNOut(xtest,net_last)
    #save results
    ymodel_last=np.concatenate((ytest_last, ypred_last))

    #Best nn found
    #------------------------------------------------------
    net_best=nn_dict[min_err_rmse_ind]
    ytest_best = pyrenn.NNOut(xtrain,net_best)
    ypred_best = pyrenn.NNOut(xtest,net_best)
    ymodel_best=np.concatenate((ytest_best, ypred_best))
    #------------------------------------------------------

    #Save neural network
#    output_path='../data/trained_nns/'
    pyrenn.saveNN(net_best, output_path + 'NN_weights_no_overfit_' + function + '_sigma_' + str(sigma) + '_rep_' + str(n) + '_r_' + str(realization) + '.csv')

    

    xplot=np.concatenate((xtrain,xtest))

    #Figure settings                                                 
    #--------------------------------
    name_fig='no_overfit_prediction'+ '_sigma_' + str(sigma) + '_' + str(function) + '_' + str(n) + '_r_' + str(realization)

    #Define figure size
    cm = 1/2.54 #convert inch to cm
    width = 10*cm; height=8*cm 

    #Fonts and sizes
    size_axis=7;size_ticks=6;size_title=5
    line_w=1;marker_s=3
    #--------------------------------


    fig=figure(figsize=(width,height), dpi=300)
    plt.axvline(x=train_border,linestyle='--',linewidth=line_w, color='k')
    plt.axvline(x=valid_border,linestyle='--',linewidth=line_w, color='k')
    
    plt.plot(xplot,ymodel_n,'k', label='first nn model')
    plt.plot(xplot,ymodel_last,'r', label='last nn model')
    plt.plot(xplot,ymodel_best,'green',label='best nn model') 
    plt.plot(xplot,dn.y_noise, 'orange', label='noise')
    plt.plot(xplot,dn.y, '.', color= 'blue', label='original')


    plt.title('%s, n=%d' % (function, n),fontsize=size_title)
    plt.xlabel('x',fontsize=size_axis);plt.ylabel('y',fontsize=size_axis)
    plt.xticks(fontsize=size_ticks);plt.yticks(fontsize=size_ticks)
    plt.xlim(-2,2);plt.ylim(-0.1,1.1)
    plt.legend(loc='best', fontsize=size_ticks)
    
    plt.savefig(output_path+name_fig+ext,dpi=300)

    try:
        ymodel=np.append(ymodel,ymodel_best)
    except NameError:
        ymodel=ymodel_best


#Add predictions to data
d['ymodel']=ymodel

#Save updated data with model
d.to_csv( output_path + 'NN_no_overfit_' + function + '_sigma_' + str(sigma) + '_r_' + str(realization) + '.csv')
