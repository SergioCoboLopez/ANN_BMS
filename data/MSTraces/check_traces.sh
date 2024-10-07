#Show number of iterations of a batch of traces

n="$1"
function="$2" #tanh, leaky_ReLU
sigma="$3"


for i in BMS_${function}_n_${n}_sigma_${sigma}_r_{0..2}_trace_50000_prior_10.csv;
do echo $i; tail -n1 $i | cut -d';' -f1;
   done
