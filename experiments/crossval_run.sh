
for FOLD in 0 1 2 3 4 5 6 7
do
    HYDRA_FULL_ERROR=1 python train.py experiment.fold_idx=$FOLD experiment.grouped=1 experiment.dataset=ATEPP experiment.symrep=sequence experiment.input_format=perfmidi experiment.lr=1e-6 sequence.mid_encoding=CPWord experiment.device=1 experiment.batch_size=1 sequence.BPE=0 sequence.max_seq_len=1600 experiment.no_ae_run=True
    # HYDRA_FULL_ERROR=1 python train.py experiment.fold_idx=$FOLD experiment.grouped=1 experiment.dataset=ASAP experiment.symrep=graph experiment.input_format=perfmidi experiment.batch_size=32 experiment.lr=0.0001 experiment.device=1 graph.bi_dir=True graph.homo=0 experiment.no_ae_run=True
    # HYDRA_FULL_ERROR=1 python train.py experiment.fold_idx=$FOLD experiment.grouped=1 experiment.dataset=ASAP experiment.task=difficulty_id experiment.symrep=matrix experiment.input_format=perfmidi experiment.lr=3e-5 experiment.lr_gamma=0.99 experiment.device=5 experiment.batch_size=16 matrix.resolution=800 experiment.no_ae_run=True 
donegpu