"""
This script will run all the experiments for ITC {topology_name} 11 nodes topologie
We will vary the following  parameters:
    traffic matrix : [0, 1, 2, 3]
    sync step : [1, 2, 3, 4, 5, 6, 7, 8, 9]
    agent_type : ["dqn_model_sharing", "digital-twin", "dqn_value_sharing"]
    dqn model : ["original", "light", "lighter", "lighter_2", "lighter_3", "ff"]
    
"""


def generate_command(
    train,
    sim_duration,
    session_name,
    traffic_matrix_index,
    agent_type,
    topology_name,
    experiment_name,
    batch_size,
    save_models,
    learning_rate,
    exploration_initial_eps,
    exploration_final_eps,
    replay_buffer_max_size,
    load_factor,
    sync_step,
    max_out_buffer_size,
    loss_penalty_type,
    load_path,
    snapshot_interval,
    pingPacketIntervalTime,
    numEpisodes,
    saved_models_path,
    gap_threshold,
    optimal_solution_path
):
    """Generate the simulation command"""
    # simulation_command = f"python3 -u main.py --seed={seed} --simTime={sim_duration} --train={train} --basePort=7000 --agent_type={agent_type} --session_name={session_name} --agent_type={agent_type} --logs_parent_folder=examples/{topology_name}/results/{experiment_name} --traffic_matrix_root_path=examples/{topology_name}/traffic_matrices/ --traffic_matrix_index={traffic_matrix_index} --overlay_adjacency_matrix_path=examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/{topology_name}/topology_files/node_coordinates.txt --map_overlay_path=examples/{topology_name}/topology_files/map_overlay.txt --training_step=0.01 --batch_size={batch_size} --lr={learning_rate} --exploration_final_eps={exploration_final_eps} --exploration_initial_eps={exploration_initial_eps} --iterationNum=5000 --gamma=1.0 --save_models={save_models} --start_tensorboard=0 --replay_buffer_max_size={replay_buffer_max_size} --link_delay=1 --load_factor={load_factor} --sync_step={sync_step} --max_out_buffer_size={max_out_buffer_size} --sync_ratio=0.2 --signalingSim=1  --prioritizedReplayBuffer={prioritizedReplayBuffer} --activateUnderlayTraffic={activateUnderlayTraffic} --bigSignalingSize={bigSignalingSize} --groundTruthFrequence=1 --pingAsObs=1 --load_path={load_path} --loss_penalty_type={loss_penalty_type} --snapshot_interval={snapshot_interval} --smart_exploration={smart_exploration} --lambda_train_step={lambda_train_step} --buffer_soft_limit={buffer_soft_limit} --lambda_lr={lambda_lr} --lamda_training_start_time={lamda_training_start_time}--pingPacketIntervalTime={pingPacketIntervalTime} --numEpisodes={numEpisodes}--reset_exploration={reset_exploration}--tunnels_max_delays_file_name=examples/{topology_name}/topology_files/max_observed_values.txt --saved_models_path={saved_models_path} --gap_threshold={gap_threshold} --packet_size=516"
    simulation_command = f"python3 main.py --agent_type {agent_type} --nb_episodes {numEpisodes} --train {train} --activateUnderlayTraffic 0 --map_overlay_path examples/{topology_name}/topology_files/overlay_map.txt --load_factor {load_factor} --physical_adjacency_matrix_path examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --overlay_adjacency_matrix_path examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --tunnels_max_delays_file_name examples/{topology_name}/topology_files/max_observed_values.txt --traffic_matrix_path examples/{topology_name}/traffic_matrices/node_intensity_normalized_{traffic_matrix_index}.txt --node_coordinates_path examples/{topology_name}/topology_files/node_coordinates.txt --logs_parent_folder examples/{topology_name}/results/{experiment_name} --start_tensorboard 0 --tensorboard_port 16666 --batch_size {batch_size} --episode_duration {sim_duration} --pingAsObs 0 --logging_timestep 1 --pingPacketIntervalTime {pingPacketIntervalTime} --gap_threshold {gap_threshold} --training_step 0.05 --target_update_period {sync_step} --exploration_initial_eps {exploration_initial_eps} --exploration_final_eps {exploration_final_eps} --save_models {save_models} --lr {learning_rate} --load_path {load_path} --replay_buffer_max_size {replay_buffer_max_size} --loss_penalty_type {loss_penalty_type} --session_name {session_name} --max_out_buffer_size {max_out_buffer_size} --saved_models_path {saved_models_path} --optimal_solution_path {optimal_solution_path}"
    return simulation_command


import os
import subprocess

# static parameters
traff_mats = list(range(0, 4))
# traff_mats = [0, 1, 2, 3]
# traff_mats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
# traff_mats = [0,]

sync_steps = list(range(1, 10)) + list(range(10, 22, 5))
# sync_steps = [1, 2, 4]
sync_steps = list(range(1, 10, 2))
sync_steps = [1, 5 , 10]
sync_steps = [1]

train_loads = [0.4]
# train_loads = [0.9]

agent_types = ["others", ]
# agent_types= [ "other"]
# agent_types= ["dqn_value_sharing",]
# agent_types = [""]
# agent_types = ["ideal"]
# agent_types = ["ideal"]

loss_pen_types = ["fixed"]
# loss_pen_types = ["None"]
# loss_pen_types = ["constrained"]
base_topo = "abilene"

if base_topo == "abilene":
    topology_name = "5n_overlay_full_mesh_abilene"
    topology_name = "abilene"
elif base_topo == "geant":
    topology_name = "overlay_full_mesh_10n_geant"
    topology_name = "geant"

experiment_name = f"{base_topo}_results"
# experiment_name = "test"

seed = 100
rb_size = 15000
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3"]
dqn_models = []
dqn_models = [""]
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
nn_sizes = [35328, 9728, 5120, 1536, 512]
nn_sizes = [
    35328,
]
# d_t_max_time = 10
# variable parameters
bs = 512
lr = 0.0001
explorations = [
    ["vary", 1.0, 0.01],
]
# explorations = [["vary", 1.0, 0.01]]

train_duration = 60
test_duration = 60
max_output_buffer_sizes = [
    16260,
]
lambda_waits = [0, 25, 80]
lambda_waits = [0, 80, train_duration]
lambda_waits = [0, 40]
lambda_waits = [
    0,
]
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
buffer_soft_limits = [0.6, 0.4, 0.1]
buffer_soft_limits = [
    0,
]
max_num_episodes = 1
pingPacketIntervalTimes = [0.1, 0.5, 1]
pingPacketIntervalTimes = [10000]

# thresholds =  np.arange(0.0, 1.0, 0.1)
thresholds = [0.0, 0.25, 0.5, 0.75]
thresholds =  [0.0,]
# [0.16, 0.05, 0.03, 0.01]
test_only = True
reset_exploration = 0

iii = 0


command_root = f"docker run --rm --gpus all -v /home/redha/prisma_chap4/prisma/:/app/prisma1/ -v /mnt/journal_paper_results/{base_topo}_itc/saved_models:/mnt/journal_paper_results/{base_topo}_itc/saved_models -w /app/prisma1 allicheredha/prismacopy:3.0"
inc = 0

for traff_mat in traff_mats:
    for train_load in train_loads:
        for loss_penalty_type in loss_pen_types:
            for threshold in thresholds:
                for pingPacketIntervalTime in pingPacketIntervalTimes:
                            for max_output_buffer_size in max_output_buffer_sizes:
                                for exploration in explorations:
                                        for agent_type in agent_types:
                                            if (
                                                agent_type == "dqn_model_sharing"
                                                or agent_type == "dqn_logit_sharing"
                                                or agent_type == "dqn_value_sharing"
                                            ):
                                                for idx, dqn_model in enumerate(
                                                    dqn_models
                                                ):
                                                    for sync_step in sync_steps:
                                                        if sync_step != 1 and (
                                                            agent_type
                                                            == "dqn_value_sharing"
                                                        ):
                                                            continue

                                                        session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn_{dqn_model}_{agent_type}_size_{nn_sizes[idx]}_tr_{train_load}_sim_{train_duration}_{max_num_episodes}_lr_{lr}_bs_{bs}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_{exploration[0]}_reset_{reset_exploration}_threshold_{threshold}"

                                                        # session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn_{agent_type}_size_{nn_sizes[idx]}_tr_{train_load}_sim_{train_duration}_{max_num_episodes}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_ping_{pingPacketIntervalTime}_loss_{rcpo_consider_loss}_reset_{reset_exploration}_use_loss_{rcpo_use_loss_pkts}"
                                                        saved_models_path = f"/mnt/journal_paper_results/{base_topo}_itc/saved_models/"
                                                        # saved_models_path = f"/mnt/journal_paper_results/geant_overlay/saved_models/{session_name}"
                                                        # launch training
                                                        python_command = generate_command(
                                                            train=1,
                                                            sim_duration=train_duration,
                                                            session_name=session_name,
                                                            traffic_matrix_index=traff_mat,
                                                            agent_type=agent_type,
                                                            topology_name=topology_name,
                                                            experiment_name=experiment_name,
                                                            batch_size=bs,
                                                            learning_rate=lr,
                                                            save_models=1,
                                                            exploration_initial_eps=exploration[
                                                                1
                                                            ],
                                                            exploration_final_eps=exploration[
                                                                2
                                                            ],
                                                            replay_buffer_max_size=rb_size,
                                                            load_factor=train_load,
                                                            sync_step=sync_step,
                                                            max_out_buffer_size=max_output_buffer_size,
                                                            loss_penalty_type=loss_penalty_type,
                                                            snapshot_interval=train_duration,
                                                            load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                                            numEpisodes=max_num_episodes,
                                                            saved_models_path=saved_models_path,
                                                            gap_threshold=threshold,
                                                        )

                                                        full_command = f"tsp {command_root} {python_command}"
                                                        print(full_command)

                                                        if test_only is False:
                                                            task_id = int(
                                                                subprocess.check_output(
                                                                    full_command,
                                                                    shell=True,
                                                                )
                                                            )
                                                            iii += 1
                                                            print(
                                                                task_id,
                                                                "train",
                                                                full_command,
                                                            )
                                                            # pass

                                                        # sleep(0.3)
                                                        # put the job on the top of the queue
                                                        # subprocess.check_output(f"tsp -u {task_id}", shell=True)
                                                        # launch testing for final model and intermediate models
                                                        # saved_models_path = f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                        for test_load in test_loads:
                                                            # for model_version in [
                                                            #     "final"
                                                            # ] + [
                                                            #     f"episode_{i}_step_{i}"
                                                            #     for i in range(
                                                            #         1, 20, 4
                                                            #     )
                                                            # ]:
                                                            #     if (
                                                            #         model_version
                                                            #         != "final"
                                                            #     ) and not (
                                                            #         sync_step == 1
                                                            #         and train_load
                                                            #         == 0.9
                                                            #         and loss_penalty_type
                                                            #         == "constrained"
                                                            #         and threshold
                                                            #         == 0.0
                                                            #     ):
                                                            #         continue
                                                            for model_version in [
                                                                "final"
                                                            ]:
                                                                python_command = generate_command(
                                                                    train=0,
                                                                    sim_duration=test_duration,
                                                                    session_name=str(
                                                                        session_name
                                                                    ),
                                                                    traffic_matrix_index=traff_mat,
                                                                    agent_type=agent_type,
                                                                    topology_name=topology_name,
                                                                    experiment_name=experiment_name,
                                                                    batch_size=bs,
                                                                    learning_rate=lr,
                                                                    save_models=0,
                                                                    exploration_initial_eps=exploration[
                                                                        1
                                                                    ],
                                                                    exploration_final_eps=exploration[
                                                                        2
                                                                    ],
                                                                    replay_buffer_max_size=rb_size,
                                                                    load_factor=test_load,
                                                                    sync_step=sync_step,
                                                                    max_out_buffer_size=max_output_buffer_size,
                                                                    loss_penalty_type=loss_penalty_type,
                                                                    snapshot_interval=0,
                                                                    load_path=f"{saved_models_path.rstrip('/')}/{session_name}/{model_version}",
                                                                    pingPacketIntervalTime=pingPacketIntervalTime,
                                                                    numEpisodes=1,
                                                                    saved_models_path=saved_models_path,
                                                                    gap_threshold=threshold,
                                                                )
                                                                if test_only:
                                                                    full_command = f"tsp {command_root} {python_command}"
                                                                else:
                                                                    full_command = f"tsp -D {task_id} {command_root} {python_command}"
                                                                # sleep(0.3)
                                                                print(full_command)
                                                                raise(1)
                                                                test_task_id = int(
                                                                    subprocess.check_output(
                                                                        full_command,
                                                                        shell=True,
                                                                    )
                                                                )
                                                                print(
                                                                    test_task_id,
                                                                    "test",
                                                                    test_load,
                                                                )
                                                                # put the job on the top of the queue
                                                                # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)

                                                                # sleep(2)

                                            else:
                                                for model in ["oracle_routing"]:
                                                    for test_load in test_loads:
                                                        session_name = f"{model}_{traff_mat}_{test_duration}"
                                                        python_command = generate_command(
                                                            train=0,
                                                            sim_duration=test_duration,
                                                            session_name=str(
                                                                session_name
                                                            ),
                                                            traffic_matrix_index=traff_mat,
                                                            agent_type=model,
                                                            topology_name=topology_name,
                                                            experiment_name=experiment_name,
                                                            batch_size=bs,
                                                            learning_rate=lr,
                                                            save_models=0,
                                                            exploration_initial_eps=exploration[
                                                                1
                                                            ],
                                                            exploration_final_eps=exploration[
                                                                2
                                                            ],
                                                            replay_buffer_max_size=rb_size,
                                                            load_factor=test_load,
                                                            sync_step=1,
                                                            max_out_buffer_size=max_output_buffer_size,
                                                            loss_penalty_type="fixed",
                                                            snapshot_interval=0,
                                                            load_path=None,
                                                            pingPacketIntervalTime=pingPacketIntervalTime,
                                                            numEpisodes=1,
                                                            saved_models_path=None,
                                                            gap_threshold=0.0,
                                                            optimal_solution_path=f"examples/abilene/optimal_solution/{traff_mat}_norm_matrix_uniform/{int(test_load*100)}_ut_minCostMCF.json"
                                                        )

                                                        # if test_load == 0.6:
                                                            # if os.path.exists(
                                                            #     f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                            # ):
                                                            #     os.system(
                                                            #         f"rm -rf examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                            #     )
                                                        full_command = f"tsp {command_root} {python_command}"
                                                        # else:
                                                        #     full_command = f"tsp -D {test_task_id} {command_root} {python_command}"
                                                        print(full_command)
                                                        test_task_id = int(
                                                            subprocess.check_output(
                                                                full_command,
                                                                shell=True,
                                                            )
                                                        )
                                                        # raise(3)
                                                        # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)
                                                        print(
                                                            test_task_id,
                                                            "test",
                                                            test_load,
                                                        )
                                                inc += 1
    print(iii)
