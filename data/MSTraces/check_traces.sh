
#Show number of iterations of a batch of traces

function="$1"

for i in {0..9};
do echo $i ; tail -n1 BMS_long_${function}_trace.${i}_prior_10.csv | cut -d';' -f1;
   done
