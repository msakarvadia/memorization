#for file in *_5.pbs; do qsub $file; done
#for file in *_3.pbs; do qsub $file; done

for file in *.pbs; do qsub $file; done
