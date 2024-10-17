

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
#-------------------------------
in_path='../data/'

resolution=0.01
function='tanh' #'tanh', 'leaky_ReLU'
file1=in_path +'NN_function_'+ str(function) + '_NREP_10_data.csv'
file2=in_path +'NN_function_'+ str(function) + '_NREP_10_' + 'res_' + str(resolution) + '_data.csv'

n=0

d1=pd.read_csv(file1); d1=d1[d1['rep']==n]
d2=pd.read_csv(file2); d2=d2[d2['rep']==n]
#-------------------------------

#plot
#-------------------------------

#Define figure size in cm
cm = 1/2.54 #convert inch to cm
width = 8*cm; height=4*cm


#Figure settings                                                     
#--------------------------------                                    
output_path='../results/figures/'
name_fig='example_fig'
extensions=['.svg','.png','.pdf']     #Extensions to save figure     

#Define figure size                                                  
cm = 1/2.54 #convert inch to cm                                      
width = 8*cm; height=4*cm #8x4cm for each figure in panel

#Fonts and sizes                                                     
size_axis=7;size_ticks=6;size_title=5
line_w=1;marker_s=3
#--------------------------------

#Plots                                                  
#--------------------------------                                    
plt.plot(d1.x1,d1.y,'.',markersize=marker_s, color='blue', label='low res')
plt.plot(d2.x1,d2.y,'.',markersize=marker_s, color='red', label='high res')


line_w=1;marker_s=1 #width and marker size



#Labels                                                              
plt.xlabel('x',fontsize=size_axis);plt.ylabel('y',fontsize=size_axis)

#Ticks
xmin=np.min(d1.x1)


# x_step=5
# xtick_labels=[tick for tick in d1.x ]
# plt.xticks(xtick_labels, fontsize=size_ticks)

# y_step=100
# ytick_labels=[tick for tick in range(0,450,y_step) ]
# plt.yticks(ytick_labels, fontsize=size_ticks)

#legend                                                              
plt.legend(loc='best',fontsize=size_ticks,frameon=False)

#save fig                                                            
# for ext in extensions:
#     plt.savefig(output_path+name_fig+ext,dpi=300)

plt.show()
#-------------------------------- 

