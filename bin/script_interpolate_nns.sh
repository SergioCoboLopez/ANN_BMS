#!/bin/bash

declare -a functions=('tanh'  'leaky_ReLU')
declare -a sigmas=('0.0' '0.02' '0.04'  '0.06' '0.08' '0.1' '0.12' '0.14' '0.16' '0.18' '0.2')


for s in "${sigmas[@]}";
      do for f in "${functions[@]}";
      do for r in {0..2};
	    do echo $f $s $r;
            python3 interpolate_nns.py $f $s $r;
	    done
	 done
	    done
