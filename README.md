# Bayesian Machine Scientist vs Artificial Neural Netkorks

## Can a Bayesian Machine Scientist beat Artificial Neural Networks?

Step 1 - Generate 20 functions using Neural Networks with the following architecture:
   1.1. Input Layer Size: 1
   1.2. Number of layers: 5
   1.3. Layer size: 10
   1.4. Activation functions: hyperbolic tangent (10 functions) and Leaky ReLU (10 functions)

The functions are normalized ($$y=[0,1]$$) and data range from $$x=[-4,4]$$. We take the subset $$x_1=[-2,2]$$.

The code for generating these data is 'generate_ANN_data'. This code is meant to be used a single time, because it generates the benchmark data to compare ANNs and the BMS. The functions are saved as '.csv' files.

Step 2 - Add gaussian noise to data.