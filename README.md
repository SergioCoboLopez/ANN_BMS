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

We use the Levenberg-Marquadt [algorithm](https://pyrenn.readthedocs.io/en/latest/train.html) from [pyrenn](https://pyrenn.readthedocs.io/en/latest/index.html) to train neural networks.
We split each function into three sets: a train set (first 50 points), validation (10 points), and test set (20 points).
We train each neural network with two iterations of the train_LM function in pyrenn. We repeat this training 300 times (epochs), where the input for each epoch is the output of the previous one.
We track the root squared mean error of each neural network on the validation set over the 300 epochs and save the neural network with the lowest error on the validation test to prevent overfitting.


### Compare results for a single realization

The code: BMS_energy_mdl.ipynb

More information soon

### Compare overall results

The codes: BMS_vs_ANNs.ipynb and plot_BMS_vs_ANNs.ipynb

Because the operations to generate overall results are time and computationally consuming, we do the data processing on one code and the figure plotting on another one.

##Computing the sem of errors of overall calculations
For a given value of $$\sigma=\sigma_i$$ w ran 30 simulations for the tanh activation function and 30 simulations for the leaky ReLu activation function. The 30 simulations correspond to $$N=10$$ functions and $$R=3$$ noise realizations.
We plotted the mean of these 30 simulations.
As for the error, we calculated the standard error of the mean over the average of the three realizations of the noise.
