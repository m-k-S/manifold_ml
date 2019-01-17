declare -a dsnames=("karate" "football" "polbooks")
declare -a reg_vals=("0.1" "0.2" "0.3" "0.5" "0.7" "0.8" "0.9")
declare -a lmbda_vals=("0" "0.1" "0.2" "0.5" "0.7" "1" "2" "4" "10")

#method_name="mmc"
#declare -a k_vals=("0")   # use dummy k value for mmc (because it doesnt use the input k

method_name="lmnn"
declare -a k_vals=("1" "3" "5" "7" "11")

output_filename="run_pll_calls.sh"
batch_size=20

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
   echo "python3 learn-metric.py --dataset $dsn --method $method_name --K $k --reg $reg --lmbd $lmbd  &" >> $output_filename
   if ! (($CTR % $batch_size)); then
	   echo "wait" >> $output_filename
   fi
done
done
done
done

