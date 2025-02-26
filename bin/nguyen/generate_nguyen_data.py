#28/11/2024. 

#Import libraries                                                                                  
#++++++++++++++++++++++++++++++++++++++++++++                                                      
import seaborn as sns
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.pyplot as plt
from decimal import Decimal
import math
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.pyplot import figure
import copy
import sys
from matplotlib import colormaps
#++++++++++++++++++++++++++++++++++++++++++++ 

#----------------------------
def nguyen0(x):
    y1=3.39*x**3 + 2.12*x**2 +1.78*x
    return y1
#----------------------------

#----------------------------
def nguyen1(x):
    y5=np.sin(x**2)*np.cos(x) - 0.75
    return y5
#----------------------------

#----------------------------
def nguyen2(x):
    y7=np.log(x+1.4) + np.log(x**2 + 1.3)
    return y7
#----------------------------

#---------------------------
def nguyen3(x):
    y8=np.sqrt(1.23*x)
    return y8
#----------------------------

#---------------------------
def nguyen4(x,y):
    y10=np.sin(1.5*x)*np.cos(0.5*y)
    return y10
#----------------------------


#define x,y
points=80

#f1
x1_i=-1;x1_f=1; step1 = (x1_f - x1_i)/points
x1= np.arange(x1_i, x1_f, step1)

#f5
x5_i=-1;x5_f=1; step5 = (x5_f - x5_i)/points
x5 = np.arange(x5_i, x5_f, step5)

#f7
x7_i=0;x7_f=2; step7 = (x7_f - x7_i)/points
x7 = np.arange(x7_i, x7_f, step7)

#f8
x8_i=0;x8_f=4; step8 = (x8_f - x8_i)/points
x8 = np.arange(x8_i, x8_f, step8)

#f10
x10_i=0;x10_f=1; step10 = (x10_f - x10_i)/points
x10 = np.arange(x10_i, x10_f, step10)
y10 = np.arange(x10_i, x10_f, step10) #y=x, no need to redefine the vector

ydumb = [0]*len(y10) #placeholder for csv dataframes


#define z=nguyen(x,y)
z1 = [ nguyen0(x) for x in x1]
z5 = [ nguyen1(x) for x in x5]
z7 = [ nguyen2(x) for x in x7]
z8 = [ nguyen3(x) for x in x8]
z10 = [ nguyen4(x,y) for (x,y) in zip(x10,y10)]
nguyen_id=[1]*len(x1)
print(nguyen_id)

d1 = pd.DataFrame({'x' : x1, 'y': ydumb, 'z' : z1, 'rep':[1]*len(x1)})
d5 = pd.DataFrame({'x' : x5, 'y': ydumb, 'z' : z5, 'rep':[5]*len(x5)})
d7 = pd.DataFrame({'x' : x7, 'y': ydumb, 'z' : z7, 'rep':[7]*len(x7)})
d8 = pd.DataFrame({'x' : x8, 'y': ydumb, 'z' : z8, 'rep':[8]*len(x8)})
d10 = pd.DataFrame({'x': x10, 'y': y10,  'z' : z10,'rep':[10]*len(x10)})

d_all=pd.concat([d1,d5, d7,d8,d10])

print(d_all)

d_all.to_csv('../../data/generative_data/' + 'nguyen_data_all.csv')


#Define figure size in cm
#--------------------------------
cm = 1/2.54 #convert inch to cm
width = 8*cm; height=4*cm

#Fonts and sizes                                                     
size_axis=7;size_ticks=6;size_title=5
line_w=1;marker_s=3
#--------------------------------

#Plots                                                               
#--------------------------------
#Figure Size
cm = 1/2.54  # centimeters in inches 
width=24*cm;height=12*cm #Width and height of plots 
matplotlib.rcParams['figure.figsize'] = [width, height]
rows=2;cols=4
#Fonts and sizes                                                     
size_axis=7;size_ticks=6;size_title=7

gs=gridspec.GridSpec(rows,cols)
gs.update(left=0.05,right=0.97,bottom=0.1,top=0.95,wspace=0.35,hspace=0.65)

#Nguyen1
ax_00=plt.subplot(gs[0,0])
plt.title(r'$y_1=3.39 x^3 + 2.12 x^2 +1.78x$',fontsize=size_title)
plt.plot(x1,z1, '.', color='blue', label='f1')
plt.xlabel('x',fontsize=size_axis)
plt.ylabel('y',fontsize=size_axis)


#Nguyen1
ax_01=plt.subplot(gs[0,1])
plt.title(r'$y_5=\sin(x^2) \cos(x)$',fontsize=size_title)
plt.plot(x5,z5, '.', color='blue', label='f1')
plt.xlabel('x',fontsize=size_axis)
plt.ylabel('y',fontsize=size_axis)


#Nguyen1
ax_02=plt.subplot(gs[0,2])
plt.title(r'$y_7=\log(x+1.4) + \log(x^2 + 1.3)$',fontsize=size_title)
plt.plot(x7,z7, '.', color='blue', label='f1')
plt.xlabel('x',fontsize=size_axis)
plt.ylabel('y',fontsize=size_axis)


#Nguyen4
ax_03=plt.subplot(gs[1,0])
plt.title(r'$y_8=\sqrt{1.23x}$',fontsize=size_title)
plt.plot(x8,z8, '.', color='blue', label='f1')
plt.xlabel('x',fontsize=size_axis)
plt.ylabel('y',fontsize=size_axis)


#Nguyen5
ax_04 = plt.subplot(gs[1:3,1], projection='3d')
plt.title(r'$y_{10}=\sin(1.5x) \cos(0.5y)$',fontsize=size_title)
ax_04.plot(x10, y10, z10, '.',color='blue', label='test')
ax_04.set_xlabel('x',fontsize=size_axis)
ax_04.set_ylabel('y',fontsize=size_axis)
ax_04.set_zlabel('z',fontsize=size_axis)


plt.savefig('../../results/' + 'nguyen_functions.png',dpi=300)
#plt.show()
