export CUDA_VISIBLE_DEVICES=0,1,2,3

username=$(whoami)
if [ -d "/data/$username" ]; then
    setlogdir="paths.log_dir=/data/$username/runs "
else
    setlogdir=""
fi

horovodrun -np 4 python run_train.py \
    model.name=parkour \
    algo.name=parkour \
    network=gnn \
    config.headless=True \
    config.multi_gpu=True \
    environment.env.enableCameraSensors=True \
    environment.env.numEnvs=2048 \
    environment.env.envSpacing=0 \
    environment.terrain.num_rows=20 \
    environment.terrain.num_cols=20 \
    environment.terrain.terrain_length=10 \
    environment.terrain.terrain_width=4 \
    environment.terrain.walk_terrain.num_rectangles=0 \
    environment.terrain.border_size=1 \
    config.amp_obs_demo_buffer_size=20000 \
    config.amp_replay_buffer_size=20000 \
    config.save_frequency=20 \
    config.task_name=train_walk \
    $setlogdir \
    # config.minibatch_size=16 \
    # config.amp_batch_size=16 \
    # config.amp_minibatch_size=16 \