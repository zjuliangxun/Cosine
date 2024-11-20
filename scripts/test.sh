export CUDA_VISIBLE_DEVICES=0 
export CUDA_LAUNCH_BLOCKING=1
python -m pdb run_train.py \
    environment.env.numEnvs=4 \
    config.task_name=test \
    config.headless=False \
    environment.terrain.num_rows=2 \
    environment.terrain.num_cols=2 \
    config.minibatch_size=16 \
    config.amp_batch_size=16 \
    config.amp_minibatch_size=16 \
