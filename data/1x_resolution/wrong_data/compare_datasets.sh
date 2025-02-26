#!/bin/bash


declare -a sigmas=("0.0" "0.02" "0.04" "0.06" "0.08" "0.10" "0.12" "0.14" "0.16" "0.18" "0.2")


for sigma in "${sigmas[@]}"
do file1=NN_tanh_sigma_${sigma}_r_1.csv file2=NN_tanh_sigma_${sigma}_r_2.csv	  
diff -q $file1 $file2 1>/dev/null
if [[ $? == "0" ]]
then
  echo "The same"
else
  echo "Not the same"
fi

done
