#Plot real data vs NN predictions, plot NN weights, and real vs predicted

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
activation_function='leaky_ReLU'
train_size=60
noise=True

#Model predictions
if noise==True:
    file_model='NN_model_noise_' + activation_function + '_train_' +str(train_size)+ '_NREP_10_data' + '.csv'
else:
    file_model='NN_model_' + activation_function + '_train_' +str(train_size)+ '_NREP_10_data' + '.csv'

    
model_d='../data/' + file_model
d=pd.read_csv(model_d)

if noise==True:
    files_weights= 'NN_weights_noise_' + activation_function + '_train_' +str(train_size) + '_rep_'
else:
    files_weights= 'NN_weights_' + activation_function + '_train_' +str(train_size) + '_rep_'
    
model_w='../data/' + files_weights


#Figure settings                                                                          
#===================================================================            
#Path to save figure                                                      
#Output_Path='../results/Figures/'
Output_Path='../results/'

if noise==True:
    Name_figure='results_noise_' + activation_function
else:
    Name_figure='results' + activation_function
    
#Extensions                              
Extensions=['.png','.pdf']

#Gridspec parameters
n_functions=int(d['rep'].max())
rows=n_functions + 1;cols=3

#Panel size  
cm = 1/2.54 #convert inch to cm 
width = 4*cm; height=4*cm #cm per each figure

margin_l=1.0*cm;margin_r=1.2*cm
margin_t=1*cm;margin_b=2*cm

margin_cols=3.75*cm
margin_rows=0*cm

panel_w=margin_l + width*cols  + margin_cols*(cols-1) + margin_r
panel_h=margin_t + height*rows + margin_rows*(rows-1) + margin_b

ratio_l=margin_l/panel_w
ratio_r=margin_r/panel_w
ratio_cols=margin_cols/panel_w
ratio_rows=margin_rows/panel_h

#Fontsizes
size_axis=7;size_ticks=6;size_title=5


#Colors and markers
cmap='RdBu';cmap_pieces= plt.get_cmap(cmap)
color1=cmap_pieces(0.1);color2=cmap_pieces(0.9)
color3=cmap_pieces(0.3);color4=cmap_pieces(0.7)

line_w=1;marker_s=3
#===================================================================


#Panel                                                                                             
#===================================================================                               
fig=figure(figsize=(panel_w,panel_h), dpi=300)
gs=gridspec.GridSpec(rows,cols)
gs.update(left=ratio_l,right= (1- ratio_r),bottom=0.1,top=0.99,wspace=ratio_cols,hspace=ratio_rows)

train_border=d.loc[train_size-1]['x1']

for row in range(rows):

    #Read model data
    dn=d[d['rep']==row]
    dn.set_index('Unnamed: 0', inplace=True)
    dn.index.name = None
    dn=dn.reset_index(drop=True)

    #Read NN weights
    dw=model_w + str(row) + '.csv'
    dw=pd.read_csv(dw,header=12) #filter only weights
    
    #First column
    #--------------------------------------
    ax_row_0=plt.subplot(gs[row,0])

    plt.plot(dn['x1'], dn['y'],label='observed')
    plt.plot(dn['x1'], dn['y_noise'],label='observed')
    plt.plot(dn.loc[:train_size-1]['x1'], dn.loc[:train_size-1]['ymodel'],color='red',linestyle='--',label='NN model train')
    plt.plot(dn[train_size:]['x1'], dn.loc[train_size:]['ymodel'],color='orange',linestyle='--',label='NN model test')
    plt.axvline(x=train_border,linestyle='--',linewidth=line_w, color='k')

    #ticks and labels
    ax_row_0.set_ylim([-0.05,1.05])
    ax_row_0.set_yticks([0,0.5,1])
    ax_row_0.set_xlabel('x',fontsize=size_axis);ax_row_0.set_ylabel('y',fontsize=size_axis)
    #--------------------------------------
    
    #Second column
    #--------------------------------------
    ax_row_1=plt.subplot(gs[row,1])

    sns.histplot(data=dw, x="w", bins=20, stat="density",color='red')

    #ticks and labels
    ax_row_1.set_xlim([-2,2]);ax_row_1.set_ylim([0,1.25])
    ax_row_1.set_yticks([0,0.5,1])
    ax_row_1.set_xlabel('weights',fontsize=size_axis);ax_row_1.set_ylabel('Density',fontsize=size_axis)
    #--------------------------------------

    #Third column
    #--------------------------------------
    ax_row_2=plt.subplot(gs[row,2])
    
    plt.plot([0,1], [0,1],linestyle='--',linewidth=line_w)
    plt.scatter(dn.loc[:train_size-1]['y'],dn.loc[:train_size-1]['ymodel'] , s=marker_s-1, color='red',label='train')
    plt.scatter(dn.loc[train_size:]['y'],dn.loc[train_size:]['ymodel'] , s=marker_s-1, color='orange',label='test')
    
    #ticks and labels
    ax_row_2.set_ylim([-0.05,1.05])
    ax_row_2.set_yticks([0,0.5,1])
    ax_row_2.set_xlabel('y',fontsize=size_axis);ax_row_2.set_ylabel('ymodel',fontsize=size_axis)
    #--------------------------------------

    #Panel ticks and labels
    #--------------------------------------
    if row<rows-1:
        ax_row_0.set_xticks([]);ax_row_1.set_xticks([]);
        ax_row_2.set_xticks([])
    else:
        ax_row_0.set_xticks([-2, 0, 2]);ax_row_1.set_xticks([-2,0,2]);
        ax_row_2.set_xticks([0,1])
        ax_row_0.legend(fontsize=size_ticks);ax_row_2.legend(fontsize=size_ticks)

    ax_row_0.tick_params(axis='both', which='major', labelsize=size_ticks)
    ax_row_1.tick_params(axis='both', which='major', labelsize=size_ticks)
    ax_row_2.tick_params(axis='both', which='major', labelsize=size_ticks)
    #--------------------------------------

    #Save
    for ext in Extensions:
        plt.savefig(Output_Path+Name_figure+'_test' +ext,dpi=300)

plt.show()


#single figure
#--------------------------------------------------------

#Figure settings                                                     
#--------------------------------                                    
output_path='figures/'
name_fig='example_fig'
extensions=['.svg','.png','.pdf']     #Extensions to save figure     

#Define figure size                                                  
cm = 1/2.54 #convert inch to cm                                      
width = 8*cm; height=4*cm #8x4cm for each figure in panel

#Fonts and sizes                                                     
size_axis=7;size_ticks=6;size_title=5
line_w=1;marker_s=3
#--------------------------------
n=3;
dn=d[d['rep']==n]

dn.set_index('Unnamed: 0', inplace=True)
dn.index.name = None
dn=dn.reset_index(drop=True)

plt.plot(dn.x1,dn.y,'.', color='blue', label='observed')
plt.plot(dn.x1,dn.y_noise, color='orange', label='noisy')
plt.plot(dn.x1,dn.ymodel, color='red', label='nn')

plt.title('n= %d ,  %s'  %(n, activation_function)) 
plt.xlabel('x',fontsize=size_axis);plt.ylabel('y',fontsize=size_axis)
plt.xlim(-2,2);plt.ylim(-0.1,1.1)
plt.legend(loc='best')

Name_figure='nn_noisy_prediction_' + activation_function + '_' + str(n)
plt.savefig('../results/' + Name_figure + '.png',dpi=300)

plt.show()
#--------------------------------------------------------
