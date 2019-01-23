#!/bin/bash
# Usage: ./gen_pll_calls.sh DATASETNAME TEST
# TEST is "clus" or "clf"
# Runs tests for varying parameters in batches of 10

declare -a dsnames=("$1")
declare -a reg_vals=("0.2" "0.3" "0.5" "0.8")
declare -a lmbda_vals=("0" "0.1" "0.2" "0.5" "0.7" "1" "2" "4" "10")

if [ "$2" = "clus" ]; then
  declare -a k_vals=("0")   # use dummy k value for clus (because it doesnt use the input k)
fi

if [ "$2" = "clf" ]; then
  declare -a k_vals=("1" "3" "5" "7" "11")
fi

output_filename="run_pll_calls.sh"
batch_size=10

#####################################################################################################

echo "#pll calls" > $output_filename

CTR=0
for dsn in "${dsnames[@]}"
do
for k in "${k_vals[@]}"
do
for reg in "${reg_vals[@]}"
do
for lmbd in "${lmbda_vals[@]}"
do
   let CTR=CTR+1
   echo "python3 metric_learning.py --dataset $dsn --$2 --K $k --reg $reg --lmbd $lmbd  &" >> $output_filename
   if ! (($CTR % $batch_size)); then
	   echo "wait" >> $output_filename
   fi
done
done
done
done
