import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.gridspec as gridspec
import ast
import sys
sys.path.append('machine-scientist/')
sys.path.append('machine-scientist/Prior/')
from mcmc import *
from parallel import *
from fit_prior import read_prior_par

n=0;function='tanh'

#Read NN data
train_size=60
file_model='NN_model_' + function + '_train_' +str(train_size)+ '_NREP_10_data' + '.csv'
model_d='../data/' + file_model
d=pd.read_csv(model_d)
dn=d[d['rep']==n]
dn.set_index('Unnamed: 0', inplace=True)
dn.index.name = None
dn=dn.reset_index(drop=True)

#Read and filter BMS data
filename='BMS_' + function + '_trace.' + str(n) + '.csv'
trace=pd.read_csv('../data/MSTraces/' + filename, sep=';', header=None, names=['t', 'H', 'expr', 'parvals', 'kk1', 'kk2','kk3'])

sample_step=50
sampled_trace=trace.iloc[::sample_step, :]
print(sampled_trace)
testmodel=sampled_trace.iloc[0]
print(testmodel)

testH, testexpr, testparvals = testmodel.H, testmodel.expr, ast.literal_eval(testmodel.parvals)

VARS = ['x1',]
x = dn[[c for c in VARS]].copy()
y = dn.y

prior_par = read_prior_par('machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np10.2017-10-18 18:07:35.089658.dat')

t = Tree(
    variables=list(x.columns),
    parameters=['a%d' % i for i in range(10)],
    x=x, y=y,
    prior_par=prior_par,
    max_size=200,
    from_string=testexpr,
)
t.set_par_values(deepcopy(testparvals))

print(dn)
dn['ybms']=t.predict(x)
print(dn)



'''
plt.plot(dn['x1'], dn['y'],label='observed')

plt.plot(dn.loc[:train_size-1]['x1'], dn.loc[:train_size-1]['ymodel'],color='red',linestyle='--',label='NN model train')

plt.plot(dn[train_size:]['x1'], dn.loc[train_size:]['ymodel'],color='orange',linestyle='--',label='NN model test')

plt.show()
'''
