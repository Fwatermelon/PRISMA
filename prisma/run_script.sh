python3 main.py \
    --seed=100 \
    --basePort=8755 \
    --train=1 \
    --session_name="abilene_test9" \
    --logs_parent_folder="examples/abilene/" \
    --traffic_matrix_path="examples/abilene/traffic_matrices/node_intensity_normalized.txt" \
    --adjacency_matrix_path="examples/abilene/adjacency_matrix.txt" \
    --node_coordinates_path="examples/abilene/node_coordinates.txt" \
    --agent_type="dqn_buffer" \
    --load_factor=0.4 \
    --start_tensorboard=1 \
    --simTime=100 \
    --signaling_type="ideal" \
    --save_models=0 \
    --exploration_initial_eps=0.35 \
    --exploration_final_eps=0.01 \
    --iterationNum=3000 \
    --batch_size=1024 \
    --sync_ratio=0.1 \
    --training_step=0.005 \
    --lr=0.001 \
    --replay_buffer_max_size=50000 \
    --link_delay="0ms" \
    --tensorboard_port=65534 \