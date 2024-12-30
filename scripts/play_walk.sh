export CUDA_VISIBLE_DEVICES=0,1,2,3

username=$(whoami)
if [ -d "/data/$username" ]; then
    setlogdir="paths.log_dir=/data/$username/runs "
else
    setlogdir=""
fi

    # train=False \
python run_train.py \
    config.headless=False \
    load_checkpoint=True \
    load_path="./runs/train_walk/2024-12-26_15-53-06/nn/Humanoid.pth" \
    model.name=parkour \
    algo.name=parkour \
    network=gnn \
    environment.env.enableCameraSensors=True \
    environment.env.numEnvs=1 \
    environment.env.envSpacing=0 \
    environment.terrain.num_rows=1 \
    environment.terrain.num_cols=1 \
    environment.terrain.terrain_length=10 \
    environment.terrain.terrain_width=4 \
    environment.terrain.walk_terrain.num_rectangles=0 \
    environment.terrain.border_size=1 \
    config.minibatch_size=16 \
    config.amp_batch_size=16 \
    config.amp_minibatch_size=16 \
    config.amp_obs_demo_buffer_size=2000 \
    config.amp_replay_buffer_size=2000 \
    config.task_name=play_walk \
    $setlogdir \