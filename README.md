# Bayesian Machine Scientist vs Artificial Neural Netkorks

## Can a Bayesian Machine Scientist beat Artificial Neural Networks?

### Generate 1d functions using Neural Networks

The architecture used to generate the functions is:

   1. Input Layer Size: 1
   1. Number of layers: 5
   1. Layer size: 10
   1. Activation functions: hyperbolic tangent (10 functions) and Leaky ReLU (10 functions)

The functions are normalized ($$y=[0,1]$$) and data range from $$x=[-4,4]$$. We take the subset $$x_1=[-2,2]$$.

The code: 'generate_ANN_data.ipynb'. This code genrates the benchmark data to compare ANNs and the BMS and is meant to be used a single time. The code has no input. The outputs are 'csv' files labeled: NN_function_\<function\>_NREP_10_data.csv, with \<function\> being $$\textit{tanh}$$, $$\textit{leaky ReLU}$$ or any other function that the user would like to implement. Each 'csv' file has three columns containing the 10 functions for each activation function: $$x_1$$, $$y$$ and $$rep$$, the latter column representing the number of functions.

### Add gaussian noise to data.

Take the data generated in the previous step and add gaussian noise with $$\sigma=[0, 0.2]$$ in steps of $$\Delta \sigma = 0.02$$ (future versions might have different intervals).

The code: add_noise_to_data.ipynb. The inputs are the '.csv' files generated in the previous step. The outputs are modified '.csv' files with an additional column representing the gaussian noise. There are as many outputs as noise realizations. In the current version of this repository, there are three realizations.

### Train and test artificial ANNs

The code: train_ann_no_overfitting.py

More information soon


### Compare results for a single realization

The code: BMS_energy_mdl.ipynb

More information soon

### Compare overall results

To be developed