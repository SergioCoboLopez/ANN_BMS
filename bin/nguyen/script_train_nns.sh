#!/bin/bash

declare -a sigmas=('0.0' '0.02' '0.04'  '0.06' '0.08' '0.1' '0.12' '0.14' '0.16' '0.18' '0.2')

for s in "${sigmas[@]}";
      do for r in {0..2};
	    do echo $s $r;
            python3 train_nns_nguyen.py $s $r;
	 done
	    done
