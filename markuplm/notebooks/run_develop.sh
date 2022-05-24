#! /bin/bash
'source /home/ubuntu/miniconda3/bin/activate' &&
'conda activate markuplmft' &&
DATASET='develop' &&
# python /data/GIT/unilm/markuplm/notebooks/Convert_CF_data_to_SWDE.py 
# --dataset $DATASET
# &&
'python /data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/pack_data.py '
'--input_swde_path /data/GIT/swde/my_data/$DATASET/my_CF_sourceCode/WAE/'
'--output_pack_path /data/GIT/swde/my_data/$DATASET/my_CF_sourceCode/wae.pickle'
# &&
# python /data/GIT/unilm/markuplm/markuplmft/fine_tuning/run_swde/prepare_data.py 
# --input_groundtruth_path /data/GIT/swde/my_data/$DATASET/my_CF_sourceCode/groundtruth
# --input_pickle_path /data/GIT/swde/my_data/$DATASET/my_CF_sourceCode/wae.pickle
# --output_data_path /data/GIT/swde/my_data/$DATASET/my_CF_processed
# &&
# python /data/GIT/unilm/markuplm/notebooks/Create_dedup_dataset.py 
# --dataset $DATASET
# &&
# python /data/GIT/unilm/markuplm/notebooks/train_and_evaluate.py
# --dataset $DATASET
# &&
# python /data/GIT/unilm/markuplm/notebooks/
# --dataset $DATASET
