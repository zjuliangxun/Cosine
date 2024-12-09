export CUDA_VISIBLE_DEVICES=0 
export CUDA_LAUNCH_BLOCKING=1
# python -m pdb run_train.py \
python run_train.py \
    environment.env.numEnvs=1 \
    config.task_name=test \
    config.headless=False \
    environment.terrain.num_rows=1 \
    environment.terrain.num_cols=1 \
    environment.env.envSpacing=0 \
    environment.terrain.border_size=1 \
    config.minibatch_size=16 \
    config.amp_batch_size=16 \
    config.amp_minibatch_size=16 \

# AMP to parkour
# model.name=parkour \
# algo.name=parkour \
# network=gnn \