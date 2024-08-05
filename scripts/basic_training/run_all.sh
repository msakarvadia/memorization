num_7=20000
length=20
max_ctx=150
for seed in 0 1 2;
do
    for size in 3000 10000 20000;
    do
        declare num_{2..5}=$size
        for data_name in mult increment;
        do
            qsub -v data_name=$data_name,num_7=$num_7,num_2=$num_2,num_3=$num_3,num_4=$num_4,num_5=$num_5,length=$length,max_ctx=$max_ctx,seed=$seed general_exp.pbs
            # qsub general_exp.pbs
        done
    done
done
