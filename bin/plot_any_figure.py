import pyrenn
import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

import ast
import sys
sys.path.append('machine-scientist/')
sys.path.append('machine-scientist/Prior/')

from mcmc import *
from parallel import *
from fit_prior import read_prior_par


#Read data                                                                    
activation_function='tanh'
train_size=60
n=1

#Model predictions                                                           
file_model='NN_model_' + activation_function + '_train_' +str(train_size)+ '_NREP_10_data' + '.csv'
model_d='../Data/' + file_model
d=pd.read_csv(model_d)
dn=d[d['rep']==n]
print(dn)
VARS = ['x1',]
x= dn[[c for c in VARS]].copy() 
y= dn.y



file_BMS='test_ANN.%s.csv' % n
model_BMS='../../MSTraces/' + file_BMS
db = pd.read_csv(model_BMS, sep=';', header=None, names=['t', 'H', 'expr', 'parvals', 'kk1', 'kk2', 'kk3'])

print(db.columns)
minrow = db[db.H == max(db.H)].iloc[0]
minH, minexpr, minparvals = minrow.H, minrow.expr, ast.literal_eval(minrow.parvals)

prior_par = read_prior_par(
    'machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np10.2017-10-18 18:07:35.089658.dat'
)
    
t = Tree(
    variables=list(x.columns),
    parameters=['a%d' % i for i in range(10)],
    x=x, y=y,
    prior_par=prior_par,
    max_size=200,
    from_string=minexpr,    
)
t.set_par_values(deepcopy(minparvals))

#Define figure size in cm
cm = 1/2.54 #convert inch to cm
width = 8*cm; height=4*cm

#Fonts and sizes 
size_axis=7;size_ticks=6;size_title=5
line_w=1;marker_s=3 #width and marker size

matplotlib.use('Agg')
#fig = plt.figure(figsize=(width, height))
fig=figure(figsize=(width,height), dpi=300)

# plt.plot(dn['x1'], dn['y'],label='observed')
# plt.plot(dn['x1'], dn['ymodel'],label='ann model')
# plt.plot(dn['x1'], dn['ymodel'],label='ann model')


plt.show()


