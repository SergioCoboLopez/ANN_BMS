#3/9/2024 Plot predictions of nn with validation set

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
n=0
train_size=50;validation_size=train_size + 10
activation_function='leaky_ReLU'
noise=True

#Model predictions 
if noise==True:
    file_weights='NN_weights_no_overfit_noise_' + activation_function + '_train_' +str(train_size)+ '_rep_' + str(n) + '.csv'
    file_model='NN_best_model_noise_no_overfit_' + activation_function + '_train_' + str(train_size) + '_NREP_10_data' + '.csv'
    
else:
    file_model='NN_weights_' + activation_function + '_train_' +str(train_size)+ '_rep_' + str(n) + '.csv'

model='../data/' + file_model

test=pd.read_csv(model) #filter only weights

print(test)


dn=test[test['rep']==n]
dn=dn.drop(columns='Unnamed: 0')
dn=dn.reset_index(drop=True)
train_border=dn[dn['rep']==0].loc[train_size-1]['x1']
valid_border=dn[dn['rep']==0].loc[validation_size-1]['x1']



print(dn)

#Figure settings                                                                                 
#--------------------------------                                                                
output_path='../results/'
name_fig='no_overfit_noise_prediction'+ str(activation_function) + '_' + str(n)

#Define figure size                                                                              
cm = 1/2.54 #convert inch to cm                                                                  
width = 10*cm; height=8*cm

#Fonts and sizes  
size_axis=7;size_ticks=6;size_title=5
line_w=1;marker_s=3
#------------------------------------

fig=figure(figsize=(width,height), dpi=300)

ext='.png'

plt.axvline(x=train_border,linestyle='--',linewidth=line_w, color='k')
plt.axvline(x=valid_border,linestyle='--',linewidth=line_w, color='k')

plt.plot(dn.x1,dn.ymodel,'r', label='nn model')
# plt.plot(dn.x1,ymodel_last,'r', label='last nn model')
# plt.plot(dn.x1,ymodel_best,'green',label='best nn model')
plt.plot(dn.x1,dn.y_noise, 'orange', label='noise')
plt.plot(dn.x1,dn.y, '.', color= 'blue', label='original')


plt.title('%s, n=%d' % (activation_function, n),fontsize=size_title)
plt.xlabel('x',fontsize=size_axis);plt.ylabel('y',fontsize=size_axis)
plt.xticks(fontsize=size_ticks);plt.yticks(fontsize=size_ticks)
plt.legend(loc='best', fontsize=size_ticks)
plt.savefig(output_path+name_fig+ext,dpi=300)
plt.show()


