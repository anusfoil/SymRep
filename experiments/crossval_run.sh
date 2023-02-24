
for FOLD in 0 1 2 3 4 5 6 7
do
    HYDRA_FULL_ERROR=1 python train.py experiment.fold_idx=$FOLD experiment.grouped=1
done