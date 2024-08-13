
#data_name=$1
num_7=20000
#num_2=$3
#num_3=$4
#num_4=$5
#num_5=$6
length=20
max_ctx=150
#seed=$9
for seed in 0 1 2;
do
    for size in 3000 10000 20000;
    do
        declare num_{2..5}=$size
        for data_name in mult increment;
        do
             $data_name $num_7 $num_2 $num_3 $num_4 $num_5 $length $max_ctx $seed
        done
    done
done
