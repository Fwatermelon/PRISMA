"""
This script will run all the experiments for ITC {topology_name} 11 nodes topologie
We will vary the following  parameters:
    traffic matrix : [0, 1, 2, 3]
    sync step : [1, 2, 3, 4, 5, 6, 7, 8, 9]
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
    signaling_type : ["NN", "digital-twin", "target"]
    dqn model : ["original", "light", "lighter", "lighter_2", "lighter_3", "ff"]
    
"""
def generate_command(seed,
                     train,
                     sim_duration,
                     agent_type,
                     session_name,
                     traffic_matrix_index,
                     signaling_type, 
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
                     movingAverageObsSize,
                     prioritizedReplayBuffer,
                     activateUnderlayTraffic,
                     bigSignalingSize,
<<<<<<< HEAD
                     loss_penalty_type,
                     smart_exploration,
                     load_path,
                     snapshot_interval,
                     lambda_train_step,
                     buffer_soft_limit,
                     lambda_lr,
                     lamda_training_start_time,
                     pingPacketIntervalTime,
                     d_t_max_time,
                     numEpisodes,
                     d_t_send_all_destinations,
                     rcpo_consider_loss,
                     reset_exploration,
                     rcpo_use_loss_pkts,
                     saved_models_path,
                     gap_threshold                                 
                  ):
    """ Generate the simulation command
    """
    simulation_command = f'python3 -u main.py --seed={seed} --simTime={sim_duration} --train={train} --basePort=7000 --agent_type={agent_type} --session_name={session_name} --signaling_type={signaling_type} --logs_parent_folder=examples/{topology_name}/results/{experiment_name} --traffic_matrix_root_path=examples/{topology_name}/traffic_matrices/ --traffic_matrix_index={traffic_matrix_index} --overlay_adjacency_matrix_path=examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/{topology_name}/topology_files/node_coordinates.txt --map_overlay_path=examples/{topology_name}/topology_files/map_overlay.txt --training_step=0.01 --batch_size={batch_size} --lr={learning_rate} --exploration_final_eps={exploration_final_eps} --exploration_initial_eps={exploration_initial_eps} --iterationNum=5000 --gamma=1.0 --save_models={save_models} --start_tensorboard=0 --replay_buffer_max_size={replay_buffer_max_size} --link_delay=1 --load_factor={load_factor} --sync_step={sync_step} --max_out_buffer_size={max_out_buffer_size} --sync_ratio=0.2 --signalingSim=1 --movingAverageObsSize={movingAverageObsSize} --prioritizedReplayBuffer={prioritizedReplayBuffer} --activateUnderlayTraffic={activateUnderlayTraffic} --bigSignalingSize={bigSignalingSize} --groundTruthFrequence=1 --pingAsObs=1 --load_path={load_path} --loss_penalty_type={loss_penalty_type} --snapshot_interval={snapshot_interval} --smart_exploration={smart_exploration} --lambda_train_step={lambda_train_step} --buffer_soft_limit={buffer_soft_limit} --lambda_lr={lambda_lr} --lamda_training_start_time={lamda_training_start_time} --d_t_max_time={d_t_max_time} --pingPacketIntervalTime={pingPacketIntervalTime} --numEpisodes={numEpisodes} --d_t_send_all_destinations={d_t_send_all_destinations} --rcpo_consider_loss={rcpo_consider_loss} --reset_exploration={reset_exploration} --rcpo_use_loss_pkts={rcpo_use_loss_pkts} --tunnels_max_delays_file_name=examples/{topology_name}/topology_files/max_observed_values.txt --saved_models_path={saved_models_path} --gap_threshold={gap_threshold} --packet_size=516'
=======
                     d_t_max_time,
                     d_t_agent_type,
                     loss_penalty_type,
                     smart_exploration,
                     load_path,
                     d_t_load_path,
                     snapshot_interval,
                     lambda_train_step,
                     buffer_soft_limit_ratio,
                     lambda_lr,                     
                  ):
    """ Generate the simulation command
    """
    simulation_command = f'python3 main.py --seed={seed} --simTime={sim_duration} --train={train} --basePort=7000 --agent_type={agent_type} --session_name={session_name} --signaling_type={signaling_type} --logs_parent_folder=examples/{topology_name}/results/{experiment_name} --traffic_matrix_root_path=examples/{topology_name}/traffic_matrices/ --traffic_matrix_index={traffic_matrix_index} --overlay_adjacency_matrix_path=examples/{topology_name}/topology_files/overlay_adjacency_matrix.txt --physical_adjacency_matrix_path=examples/{topology_name}/topology_files/physical_adjacency_matrix.txt --node_coordinates_path=examples/{topology_name}/topology_files/node_coordinates.txt --map_overlay_path=examples/{topology_name}/topology_files/map_overlay.txt --training_step=0.01 --batch_size={batch_size} --lr={learning_rate} --exploration_final_eps={exploration_final_eps} --exploration_initial_eps={exploration_initial_eps} --iterationNum=5000 --gamma=1.0 --save_models={save_models} --start_tensorboard=0 --replay_buffer_max_size={replay_buffer_max_size} --link_delay="1ms" --load_factor={load_factor} --sync_step={sync_step} --max_out_buffer_size={max_out_buffer_size} --sync_ratio=0.2 --signalingSim=1 --movingAverageObsSize={movingAverageObsSize} --prioritizedReplayBuffer={prioritizedReplayBuffer} --activateUnderlayTraffic={activateUnderlayTraffic} --bigSignalingSize={bigSignalingSize} --groundTruthFrequence=1 --pingAsObs=0 --load_path={load_path} --d_t_load_path={d_t_load_path} --loss_penalty_type={loss_penalty_type} --snapshot_interval={snapshot_interval} --smart_exploration={smart_exploration} --lambda_train_step={lambda_train_step} --buffer_soft_limit_ratio={buffer_soft_limit_ratio} --lambda_lr={lambda_lr}'
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
    return simulation_command
    
 
import os, subprocess
from time import sleep
# static parameters
<<<<<<< HEAD
traff_mats = list(range(4, 10))
<<<<<<< HEAD
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
# traff_mats = [0, 1, 2, 3]
# traff_mats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
# traff_mats = [0,]

sync_steps = list(range(1, 10)) + list(range(10, 22, 5))
# sync_steps = [1, 2, 4]
sync_steps = list(range(1, 10, 2))
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
sync_steps = [1, 5, 7, 9]
sync_steps = [1,]

train_loads = [0.9, 0.4, 0.7]
train_loads = [0.9]

signaling_types= [ "NN", "target", "digital_twin"]
# signaling_types= [ "digital_twin", "NN"]
signaling_types= ["digital_twin"]
# signaling_types = [""]
# signaling_types = ["ideal"]
# signaling_types = ["ideal"]

loss_pen_types = [ "fixed", "None", "constrained"]
# loss_pen_types = ["constrained"]
base_topo = "geant"

if base_topo == "abilene":
    topology_name = "5n_overlay_full_mesh_abilene"
elif base_topo == "geant":
    topology_name = f"overlay_full_mesh_10n_geant"

experiment_name = "geant_results_with_threshold"
# experiment_name = "test"

seed = 100
rb_size = 10000
<<<<<<< HEAD
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3"]
dqn_models = []
dqn_models = [""]
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
nn_sizes = [35328, 9728, 5120, 1536, 512]
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
nn_sizes = [35328,]
# variable parameters
bs = 512
lr = 0.00001
explorations = [["vary", 1.0, 0.01],]
# explorations = [["vary", 1.0, 0.01]]
smart_explorations = [0,]
mv_avg_interval = 5
train_duration = 20
test_duration = 25
max_output_buffer_sizes = [16260,]
lambda_waits = [0, 25, 80]
lambda_waits = [0, 80, train_duration]
lambda_waits = [0, 40]
lambda_waits = [0,]
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
buffer_soft_limits = [0.6, 0.4, 0.1]
buffer_soft_limits = [0,]
lambda_train_steps = [1, 5, 10]
lambda_train_step = -1
max_num_episodes = 20
pingPacketIntervalTimes = [0.1, 0.5, 1]
pingPacketIntervalTimes = [0.1]
import numpy as np
# thresholds =  np.arange(0.0, 1.0, 0.1)
thresholds =  [0.0, 0.25, 0.5, 0.75]
thresholds =  [0.0,]
# [0.16, 0.05, 0.03, 0.01]
d_m_max_time = 3
d_t_send_all_destinations_list = [0, ]
test_only = True
rcpo_consider_loss=1
reset_exploration = 0
rcpo_use_loss_pkts=1
<<<<<<< HEAD
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536

iii = 0


<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
command_root = f"docker run --rm --gpus all -v /mnt/backup_examples_new:/app/prisma/examples -v /mnt/journal_paper_results/{base_topo}_overlay/saved_models:/mnt/journal_paper_results/{base_topo}_overlay/saved_models -w /app/prisma allicheredha/prismacopy_episodes:offband_new"
inc = 0

for traff_mat in traff_mats:
    for lambda_wait in lambda_waits:
        for train_load in train_loads:
            for loss_penalty_type in loss_pen_types:
                for threshold in thresholds:
                    for pingPacketIntervalTime in pingPacketIntervalTimes:
                        for d_t_send_all_destinations in d_t_send_all_destinations_list:
                            for buffer_soft_limit in buffer_soft_limits:
                                if loss_penalty_type != "constrained":
                                    if not(lambda_train_step == -1 and buffer_soft_limit == 0 and lambda_wait==0):
                                        continue
                                else:
                                    if lambda_wait == 1000 and (buffer_soft_limit!=1):
                                        continue
                                for max_output_buffer_size in max_output_buffer_sizes:
                                    for exploration in explorations:
                                        for smart_exploration in smart_explorations:
                                            for signaling_type in signaling_types:
                                                if signaling_type == "NN" or signaling_type == "digital_twin" or signaling_type == "target":
                                                    for idx, dqn_model in enumerate(dqn_models):
                                                        for sync_step in sync_steps:
                                                                if sync_step != 1 and (signaling_type == "digital_twin" or signaling_type == "target"):
                                                                    continue
                                                                if sync_step > 9 and (train_load != 0.9 or signaling_type != "NN"):
                                                                    continue
                                                                if sync_step in (1,3) and signaling_type == "NN" :
                                                                    continue
                                                                # if signaling_type != "digital_twin" and loss_penalty_type != "constrained":
                                                                    # continue
                                                                if signaling_type =="digital_twin" and threshold != 0.0 and loss_penalty_type != "constrained":
                                                                    continue
                                                                if signaling_type == "NN" and threshold != 0.0:
                                                                    continue
                                                                session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn_{dqn_model}_{signaling_type}_size_{nn_sizes[idx]}_tr_{train_load}_sim_{train_duration}_{max_num_episodes}_lr_{lr}_bs_{bs}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_lambda_step_{lambda_train_step}_ratio_{buffer_soft_limit}_wait_{lambda_wait}_lambda_lr_1e6_dt_time_{d_m_max_time}_ping_{pingPacketIntervalTime}_{exploration[0]}_{['one', 'multi'][d_t_send_all_destinations]}_explo_first_loss_{rcpo_consider_loss}_reset_{reset_exploration}_use_loss_{rcpo_use_loss_pkts}_threshold_{threshold}"
                                                                # session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn_{signaling_type}_size_{nn_sizes[idx]}_tr_{train_load}_sim_{train_duration}_{max_num_episodes}_outb_{max_output_buffer_size}_losspen_{loss_penalty_type}_ping_{pingPacketIntervalTime}_loss_{rcpo_consider_loss}_reset_{reset_exploration}_use_loss_{rcpo_use_loss_pkts}"
                                                                saved_models_path = f"/mnt/journal_paper_results/{base_topo}_overlay/saved_models/"
                                                                # saved_models_path = f"/mnt/journal_paper_results/geant_overlay/saved_models/{session_name}"
                                                                # launch training
                                                                python_command = generate_command(seed=seed,
                                                                                                train=1,
                                                                                                sim_duration=train_duration,
                                                                                                agent_type=f"dqn_buffer{dqn_model}",
                                                                                                session_name=session_name,
                                                                                                traffic_matrix_index=traff_mat,
                                                                                                signaling_type=signaling_type, 
                                                                                                topology_name=topology_name,
                                                                                                experiment_name=experiment_name,
                                                                                                batch_size=bs,
                                                                                                learning_rate=lr,
                                                                                                save_models=1,
                                                                                                exploration_initial_eps=exploration[1],
                                                                                                exploration_final_eps=exploration[2],
                                                                                                replay_buffer_max_size=rb_size,
                                                                                                load_factor=train_load,
                                                                                                sync_step=sync_step,
                                                                                                max_out_buffer_size=max_output_buffer_size,
                                                                                                movingAverageObsSize=mv_avg_interval,
                                                                                                prioritizedReplayBuffer=0,
                                                                                                activateUnderlayTraffic=1,
                                                                                                bigSignalingSize=nn_sizes[idx],
                                                                                                loss_penalty_type=loss_penalty_type,
                                                                                                snapshot_interval=train_duration,
                                                                                                smart_exploration=smart_exploration,
                                                                                                load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                                                                lambda_train_step=lambda_train_step,
                                                                                                buffer_soft_limit=buffer_soft_limit,
                                                                                                lambda_lr=lr*0.1,
                                                                                                pingPacketIntervalTime=pingPacketIntervalTime,
                                                                                                lamda_training_start_time=lambda_wait,
                                                                                                numEpisodes=max_num_episodes,
                                                                                                d_t_max_time=d_m_max_time,
                                                                                                d_t_send_all_destinations=d_t_send_all_destinations,
                                                                                                rcpo_consider_loss=rcpo_consider_loss,
                                                                                                reset_exploration=reset_exploration,
                                                                                                rcpo_use_loss_pkts=rcpo_use_loss_pkts,
                                                                                                saved_models_path=saved_models_path,
                                                                                                gap_threshold=threshold
                                                                                                )
                                                                                        
                                                                full_command = f'tsp {command_root} {python_command}'
                                                                print(session_name)
                                                                # raise(0)

                                                                if not test_only:
                                                                    task_id = int(subprocess.check_output(full_command, shell=True))
                                                                    iii +=1
                                                                    # task_id = 0
                                                                    print(task_id, "train", full_command)
                                                                    # pass
                                                                
                                                                # sleep(0.3)
                                                                # put the job on the top of the queue
                                                                # subprocess.check_output(f"tsp -u {task_id}", shell=True)
                                                                # launch testing for final model and intermediate models
                                                                # saved_models_path = f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                                for test_load in test_loads:
                                                                    for model_version in [f"episode_{i}_step_{i}" for i in [1, 2, 3, 4, 5, 6, 7, 10, 15, 19]]:
                                                                        if (model_version != "final") and not (train_load == 0.9 and threshold==0.0 and signaling_type == "digital_twin"):
                                                                            continue
                                                                    # for model_version in ["final"]:
                                                                        python_command = generate_command(seed=seed,
                                                                                                        train=0,
                                                                                                        sim_duration=test_duration,
                                                                                                        agent_type=f"dqn_buffer{dqn_model}",
                                                                                                        session_name=str(session_name),
                                                                                                        traffic_matrix_index=traff_mat,
                                                                                                        signaling_type="ideal", 
                                                                                                        topology_name=topology_name,
                                                                                                        experiment_name=experiment_name,
                                                                                                        batch_size=bs,
                                                                                                        learning_rate=lr,
                                                                                                        save_models=0,
                                                                                                        exploration_initial_eps=exploration[1],
                                                                                                        exploration_final_eps=exploration[2],
                                                                                                        replay_buffer_max_size=rb_size,
                                                                                                        load_factor=test_load,
                                                                                                        sync_step=sync_step,
                                                                                                        max_out_buffer_size=max_output_buffer_size,
                                                                                                        movingAverageObsSize=mv_avg_interval,
                                                                                                        prioritizedReplayBuffer=0,
                                                                                                        activateUnderlayTraffic=1,
                                                                                                        bigSignalingSize=nn_sizes[idx],
                                                                                                        loss_penalty_type=loss_penalty_type,
                                                                                                        snapshot_interval=0,
                                                                                                        smart_exploration=smart_exploration,
                                                                                                        load_path=f"{saved_models_path}/{session_name}/{model_version}",
                                                                                                        lambda_train_step=lambda_train_step,
                                                                                                        buffer_soft_limit=buffer_soft_limit,
                                                                                                        lambda_lr=lr*0.1,
                                                                                                        lamda_training_start_time=lambda_wait,
                                                                                                        pingPacketIntervalTime=pingPacketIntervalTime,
                                                                                                        d_t_max_time=d_m_max_time,
                                                                                                        numEpisodes=1,
                                                                                                        d_t_send_all_destinations=d_t_send_all_destinations,
                                                                                                        rcpo_consider_loss=rcpo_consider_loss,
                                                                                                        reset_exploration=reset_exploration,
                                                                                                        rcpo_use_loss_pkts=rcpo_use_loss_pkts,
                                                                                                        saved_models_path=saved_models_path,
                                                                                                        gap_threshold=threshold
                                                                                                        )
                                                                        if test_only:
                                                                            full_command = f'tsp {command_root} {python_command}'
                                                                        else:   
                                                                            full_command = f'tsp -D {task_id} {command_root} {python_command}'
                                                                        # sleep(0.3)
                                                                        # raise(1)
                                                                        print(full_command)
                                                                        test_task_id = int(subprocess.check_output(full_command, shell=True))
                                                                        # print(test_task_id, "test", test_load)
                                                                        # test_task_id= 0
                                                                        # put the job on the top of the queue
                                                                        # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)

                                                                        # sleep(2)
                                                                
                                                                    
                                                else:
                                                    if signaling_type == "ideal":
                                                        for model in ["sp", "opt"]:
                                                            sync_step = 1
                                                            for test_load in test_loads:
                                                                session_name = f"{model}_{traff_mat}_{test_duration}_final_ping_{1000}"
                                                                python_command = generate_command(seed=seed,
                                                                                                train=0,
                                                                                                sim_duration=test_duration,
                                                                                                agent_type=f"{model}",
                                                                                                session_name=session_name,
                                                                                                traffic_matrix_index=traff_mat,
                                                                                                signaling_type="ideal", 
                                                                                                topology_name=topology_name,
                                                                                                experiment_name=experiment_name,
                                                                                                batch_size=bs,
                                                                                                learning_rate=lr,
                                                                                                save_models=0,
                                                                                                exploration_initial_eps=exploration[1],
                                                                                                exploration_final_eps=exploration[2],
                                                                                                replay_buffer_max_size=rb_size,
                                                                                                load_factor=test_load,
                                                                                                sync_step=sync_step,
                                                                                                max_out_buffer_size=max_output_buffer_size,
                                                                                                movingAverageObsSize=mv_avg_interval,
                                                                                                prioritizedReplayBuffer=0,
                                                                                                activateUnderlayTraffic=1,
                                                                                                bigSignalingSize=nn_sizes[0],
                                                                                                loss_penalty_type=loss_penalty_type,
                                                                                                snapshot_interval=0,
                                                                                                smart_exploration=smart_exploration,
                                                                                                load_path=None,
                                                                                                lambda_train_step=lambda_train_step,
                                                                                                buffer_soft_limit=buffer_soft_limit,
                                                                                                lambda_lr=lr*0.1,
                                                                                                lamda_training_start_time=lambda_wait,
                                                                                                pingPacketIntervalTime=1000,
                                                                                                d_t_max_time=10,
                                                                                                numEpisodes=1,
                                                                                                d_t_send_all_destinations=d_t_send_all_destinations,
                                                                                                rcpo_consider_loss=rcpo_consider_loss,
                                                                                                reset_exploration=reset_exploration,
                                                                                                rcpo_use_loss_pkts=rcpo_use_loss_pkts,
                                                                                                saved_models_path="",
                                                                                                gap_threshold=threshold
                                                                                                )
                                                                
                                                                if test_load == 0.6:
                                                                    if os.path.exists(f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"):
                                                                        os.system(f"rm -rf examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}")
                                                                    full_command = f'tsp {command_root} {python_command}'
                                                                else:
                                                                    full_command = f'tsp -D {test_task_id} {command_root} {python_command}'
                                                                print(full_command)
                                                                test_task_id = int(subprocess.check_output(full_command, shell=True))
                                                                # raise(3)
                                                                # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)
                                                                print(test_task_id, "test", test_load)
                                                    inc +=1
    print(iii)
=======
# traff_mats = [3, 2, 0, 1]
# traff_mats = [1, 0, 2, 3]
traff_mats = [0,]
# sync_steps = list(range(1, 7))
sync_steps = [1,]
seed = 100
rb_sizes = [10000,] 
# signaling_types = ["NN", "digital_twin", "target"]
signaling_types = ["NN",]
# signaling_types = ["digital_twin"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"]
dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3"]
dqn_models = [""]
experiment_name = "ITC_NN_size_variations_experiment_rcpo"
nn_sizes = [35328, 9728, 5120, 1536, 512, 1024]
nn_sizes = [35328, 9728, 5120, 1536, 512]
nn_sizes = [35328]
d_t_max_time = 10
topology_name = "abilene"
# variable parameters
training_load = 0.4
bs = 1024
lr = 0.0001
# explorations = [["vary", 1.0, 0.01],
#                 ["fixed", 0.1, 0.1]]
explorations = [["vary", 1.0, 0.01]]
smart_explorations = [0,]
ping_freq = 10000
mv_avg_interval = 100
train_duration = 60
test_duration = 20
max_output_buffer_sizes = [16260,]
loss_pen_types = ["constrained"]
train_load = 0.4
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
test_loads = []
buffer_soft_limit_ratios = [0.45,]
lambda_train_steps = [1,]


inc = 0
command_root = ""#"docker run --rm --gpus all -v /home/redha/PRISMA_copy/prisma/examples:/app/prisma/examples -w /app/prisma allicheredha/prismacopy:2.8.3"
for traff_mat in traff_mats:
    for rb_size in rb_sizes:
        for loss_penalty_type in loss_pen_types:
            for lambda_train_step in lambda_train_steps:
                for buffer_soft_limit_ratio in buffer_soft_limit_ratios:
                    if loss_penalty_type != "constrained":
                        if not(lambda_train_step == 1 and buffer_soft_limit_ratio == 0.6):
                            continue
                    for max_output_buffer_size in max_output_buffer_sizes:
                        for exploration in explorations:
                            for smart_exploration in smart_explorations:
                                for signaling_type in signaling_types:
                                    for idx, dqn_model in enumerate(dqn_models):
                                        if signaling_type == "NN":
                                            for sync_step in sync_steps:
                                                    session_name = f"sync_{sync_step}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{['natural', 'smart'][smart_exploration]}_output_buffer_{max_output_buffer_size}_loss_pen_{loss_penalty_type}_lambda_step_{lambda_train_step}_soft_limit_ratio_{buffer_soft_limit_ratio}_lr_1e6"
                                                    # launch training
                                                    python_command = generate_command(seed=seed,
                                                                                    train=1,
                                                                                    sim_duration=train_duration,
                                                                                    agent_type=f"dqn_buffer{dqn_model}",
                                                                                    session_name=session_name,
                                                                                    traffic_matrix_index=traff_mat,
                                                                                    signaling_type=signaling_type, 
                                                                                    topology_name=topology_name,
                                                                                    experiment_name=experiment_name,
                                                                                    batch_size=bs,
                                                                                    learning_rate=lr,
                                                                                    save_models=1,
                                                                                    exploration_initial_eps=exploration[1],
                                                                                    exploration_final_eps=exploration[2],
                                                                                    replay_buffer_max_size=rb_size,
                                                                                    load_factor=train_load,
                                                                                    sync_step=sync_step,
                                                                                    max_out_buffer_size=max_output_buffer_size,
                                                                                    nPacketsOverlay=ping_freq,
                                                                                    movingAverageObsSize=mv_avg_interval,
                                                                                    prioritizedReplayBuffer=0,
                                                                                    activateUnderlayTraffic=1,
                                                                                    bigSignalingSize=nn_sizes[idx],
                                                                                    d_t_max_time=d_t_max_time,
                                                                                    d_t_agent_type=f"dqn_buffer{dqn_model}",
                                                                                    loss_penalty_type=loss_penalty_type,
                                                                                    snapshot_interval=10,
                                                                                    smart_exploration=smart_exploration,
                                                                                    load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                                                    d_t_load_path=f"examples/{topology_name}/pre_trained_models/dqn_buffer{dqn_model}",
                                                                                    lambda_train_step=lambda_train_step,
                                                                                    buffer_soft_limit_ratio=buffer_soft_limit_ratio,
                                                                                    lambda_lr=0.000001,
                                                                                    )
                                                                            
                                                    full_command = f'{python_command}'
                                                    # print("vs code config args = ", sum([x.split("=") for x in python_command.split(" ")[3:]], []))
                                                    task_id = int(subprocess.run(full_command))
                                                    print("train", full_command)
                                                    sleep(1)
                                                    # put the job on the top of the queue
                                                    # subprocess.check_output(f"tsp -u {task_id}", shell=True)
                                                    # launch testing for final model and intermediate models
                                                    saved_models_path = f"examples/{topology_name}/results/{experiment_name}/saved_models/{session_name}"
                                                    for test_load in test_loads:
                                                        for model_version in ["final"] + [f"episode1_step_{i}" for i in range(1, 6)]:
                                                            python_command = generate_command(seed=seed,
                                                                                            train=0,
                                                                                            sim_duration=test_duration,
                                                                                            agent_type=f"dqn_buffer{dqn_model}",
                                                                                            session_name=session_name,
                                                                                            traffic_matrix_index=traff_mat,
                                                                                            signaling_type="ideal", 
                                                                                            topology_name=topology_name,
                                                                                            experiment_name=experiment_name,
                                                                                            batch_size=bs,
                                                                                            learning_rate=lr,
                                                                                            save_models=0,
                                                                                            exploration_initial_eps=exploration[1],
                                                                                            exploration_final_eps=exploration[2],
                                                                                            replay_buffer_max_size=rb_size,
                                                                                            load_factor=test_load,
                                                                                            sync_step=sync_step,
                                                                                            max_out_buffer_size=max_output_buffer_size,
                                                                                            nPacketsOverlay=ping_freq,
                                                                                            movingAverageObsSize=mv_avg_interval,
                                                                                            prioritizedReplayBuffer=0,
                                                                                            activateUnderlayTraffic=1,
                                                                                            bigSignalingSize=nn_sizes[idx],
                                                                                            d_t_max_time=d_t_max_time,
                                                                                            d_t_agent_type=f"dqn_buffer{dqn_model}",
                                                                                            loss_penalty_type=loss_penalty_type,
                                                                                            snapshot_interval=0,
                                                                                            smart_exploration=smart_exploration,
                                                                                            load_path=f"{saved_models_path}/{model_version}",
                                                                                            d_t_load_path=f"{saved_models_path}/{model_version}",
                                                                                            lambda_train_step=lambda_train_step,
                                                                                            buffer_soft_limit_ratio=buffer_soft_limit_ratio,
                                                                                            lambda_lr=0.000001,
                                                                                            
                                                                                            )
                                                            full_command = f'tsp -D {task_id} {command_root} {python_command}'
                                                            test_task_id = int(subprocess.check_output(full_command, shell=True))
                                                            print(task_id, "test", test_load)
                                                            # put the job on the top of the queue
                                                            # subprocess.check_output(f"tsp -u {test_task_id}", shell=True)

                                                            sleep(2)
                                                    
                                                        
                                        else:
                                            sync_step = 1
                                            session_name = f"sync_{sync_step}_seed_{seed}_traff_mat_{traff_mat}_dqn_buffer{dqn_model}_{signaling_type}_ping_freq_{ping_freq}_rb_size_{rb_size}_{topology_name}_train_load_{training_load}_simTime_{train_duration}_mv_avg_{mv_avg_interval}_lr_{lr}_bs_{bs}_explo_{exploration[0]}_{d_t_max_time}_{nn_sizes[idx]}"
                                            os.system(f'tsp {command_root} ./scripts/run_geant.sh {sync_step} {seed} {traff_mat} dqn_buffer{dqn_model} {signaling_type} {ping_freq} {rb_size} {inc} {topology_name} 0 {training_load} 1 {train_duration} "{session_name}" {experiment_name} {mv_avg_interval} {lr} {bs} {exploration[1]} {exploration[2]} {d_t_max_time} {nn_sizes[idx]} dqn_buffer{dqn_model}')
                                            sleep(0.5)
                                            inc +=1
    # print(inc)
>>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
<<<<<<< HEAD
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
=======
>>>>>>> c2ae11cb7d68ce93ba53590e9f9d0b2392fbf536
