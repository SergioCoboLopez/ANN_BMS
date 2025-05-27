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
from random import sample
import random


def build_validation(all_training_points,len_validation,dataframe):

    pre_train_df=dataframe.loc[0:all_training_points-1]

    validation_df=dataframe.iloc[validation_points]
    train_df=pre_train_df.drop(labels=validation_points)

    return train_df, validation_df

#Read data
#-----------------------------------------------------
random.seed(a=1111)

sigma=sys.argv[1]
realization=int(sys.argv[2])

resolution='1x' #1x, 2x, 0.5x, 4e-3x 
resolutions={'0.5x':'0.1', '1x':'0.05', '2x': '0.025', '4e-3x':'0.004' }

input_path='../../data/nguyen/' + resolution + '_resolution/'
output_path=input_path+ 'trained_nns/'
filename=input_path + 'NN_nguyen_sigma_' + str(sigma) + '_r_' + str(realization) +  '.csv'
    
d=pd.read_csv(filename)
d=d.drop(columns='Unnamed: 0')
d=d.reset_index(drop=True)
#-----------------------------------------------------


#train/validation size 
#-----------------------------------------------------
n_nguyen=[1, 5 , 7, 8, 10]
n_points=int(len(d.index)/len(n_nguyen))
print(n_points)
pre_train_fraction=3/4;pre_train_size=int(n_points*pre_train_fraction)
print(pre_train_size)
validation_fraction=1/8;validation_size=int(n_points*validation_fraction)
validation_points=sample(range(pre_train_size), k=validation_size)
validation_points=np.sort(validation_points)

with open( output_path + 'validation_s_%s_r_%d' %(sigma, realization) + '.txt', 'a') as the_file:
    the_file.write(str(validation_points))
#-----------------------------------------------------

#Build ANN
ILS = 1;OLS=1
NL, LS = 5, 10
arch=[ILS] + NL*[LS] + [OLS]
nn=pyrenn.CreateNN(arch)

#Cross validations
iterations=300


for n in n_nguyen:
    #Read data
    dn=d[d['rep']==n]
    dn.index.name = None
    dn=dn.reset_index(drop=True)

    train_set, validation_set=build_validation(pre_train_size, validation_points, dn)

    xtrain=train_set['x']
    ytrain=train_set['z_noise']

    xvalid=validation_set['x']
    yvalid=validation_set['z_noise']

    print(xvalid)
    print(yvalid)
    # #Train NN
    # #Train on the  first points
    # xtrain = dn.loc[0:train_size-1]['x']
    # ytrain = dn.loc[0:train_size-1]['z_noise']

    # #Build validation set
    # xvalid=dn.loc[train_size:validation_size-1]['x']
    # yvalid=dn.loc[train_size:validation_size-1]['z_noise']

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
    output_path_fig='../../results/nn_w_validation_nguyen/'

    
    name_fig='validation_errors_' + 'sigma_' + str(sigma) + '_' + str(n) + '_r_' + str(realization)
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
    plt.title('n=%d' %  n,fontsize=size_title)
    ext=extensions[0]
    plt.savefig(output_path_fig+name_fig+ext,dpi=300)
    
    #First NN found
    # net_first=nn_dict[0]
    # xtest = dn.loc[train_size:]['x']
    # ytest_first = pyrenn.NNOut(xtrain,net_first)
    # ypred_first = pyrenn.NNOut(xtest,net_first)
    # #save results
    # ymodel_n=np.concatenate((ytest_first, ypred_first))

    # #Last NN found
    # net_last=nn_dict[iterations-1]
    # xtest = dn.loc[train_size:]['x']
    # ytest_last = pyrenn.NNOut(xtrain,net_last)
    # ypred_last = pyrenn.NNOut(xtest,net_last)
    # #save results
    # ymodel_last=np.concatenate((ytest_last, ypred_last))

    #Best nn found
    #------------------------------------------------------
    net_best=nn_dict[min_err_rmse_ind]
    xtest = dn.loc[pre_train_size:]['x']
    xtrain_valid=dn.loc[:pre_train_size-1]['x']
    ytest_best = pyrenn.NNOut(xtrain_valid,net_best)
    ypred_best = pyrenn.NNOut(xtest,net_best)
    ymodel_best=np.concatenate((ytest_best, ypred_best))
    
    # ytest_best = pyrenn.NNOut(xtrain,net_best)
    # ypred_best = pyrenn.NNOut(xtest,net_best)
    # ymodel_best=np.concatenate((ytest_best, ypred_best))
    #------------------------------------------------------

    #Save neural network
    pyrenn.saveNN(net_best, output_path + 'NN_weights_no_overfit_sigma_' + str(sigma) + '_rep_' + str(n) + '_r_' + str(realization) + '.csv')

    

#    xplot=np.concatenate((xtrain,xtest))

    #Figure settings                                                 
    #--------------------------------
    # name_fig='no_overfit_prediction_sigma_' + str(sigma) + '_'  + str(n) + '_r_' + str(realization)

    # #Define figure size
    # cm = 1/2.54 #convert inch to cm
    # width = 10*cm; height=8*cm 

    # #Fonts and sizes
    # size_axis=7;size_ticks=6;size_title=5
    # line_w=1;marker_s=3
    #--------------------------------


    # fig=figure(figsize=(width,height), dpi=300)
    # plt.axvline(x=train_border,linestyle='--',linewidth=line_w, color='k')
    # plt.axvline(x=valid_border,linestyle='--',linewidth=line_w, color='k')
    
    # plt.plot(xplot,ymodel_n,'k', label='first nn model')
    # plt.plot(xplot,ymodel_last,'r', label='last nn model')
    # plt.plot(xplot,ymodel_best,'green',label='best nn model') 
    # plt.plot(xplot,dn.z_noise, 'orange', label='noise')
    # plt.plot(xplot,dn.z, '.', color= 'blue', label='original')


    # print(n)
    # print(type(n))
    # plt.title('n=%d' % (n) ,fontsize=size_title)
    # plt.xlabel('x',fontsize=size_axis);plt.ylabel('y',fontsize=size_axis)
    # plt.xticks(fontsize=size_ticks);plt.yticks(fontsize=size_ticks)
    # plt.xlim(-2,2);plt.ylim(-0.1,1.1)
    # plt.legend(loc='best', fontsize=size_ticks)
    
    # plt.savefig(output_path+name_fig+ext,dpi=300)

    try:
        ymodel=np.append(ymodel,ymodel_best)
    except NameError:
        ymodel=ymodel_best


#Add predictions to data
d['zmodel']=ymodel

#Save updated data with model
d.to_csv( output_path + 'NN_no_overfit_sigma_' + str(sigma) + '_r_' + str(realization) + '.csv')

