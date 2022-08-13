#!/bin/bash
echo $1 # sync step 0.5 
echo $2 # seed
echo $3 # traff mat index
echo $4 # agent type
echo $5 # signaling type
echo $6 # n_packs
echo $7 # rb size
echo $8 # increment
echo $9 # topology
#module load singularity/3.5.2
uname -a
echo $(date)
SYNC=$(echo $1/0.5 | bc)
res0=$((${SYNC/.*} - 1))

cd ..

python3 main.py \
	--seed=$2 \
	--simTime=100 \
	--basePort=$(((4444+$2)+(1000*$3)+($res0*15)+($8*1000))) \
	--train=1 \
	--agent_type=$4 \
	--session_name="train_new_overlay_obs_in_bytes_fixed_$9_real_delay_$4_$5_ts_0_01_seed_$2_traff_mat_$3_rb_size_$7_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_refreshRate_$6_sync_$1_loss_x11_sp_init" \
	--signaling_type=$5 \
	--logs_parent_folder=examples/$9/ \
	--traffic_matrix_root_path=examples/$9/traffic_matrices/ \
	--traffic_matrix_index=$3 \
	--adjacency_matrix_path=examples/$9/adjacency_matrix.txt \
	--node_coordinates_path=examples/$9/node_coordinates.txt \
	--training_step=0.01 \
	--batch_size=512 \
	--lr=0.001 \
	--exploration_final_eps=0.01 \
	--exploration_initial_eps=1.0 \
	--iterationNum=3000 \
	--gamma=1.0 \
	--training_trigger_type="time" \
	--save_models=1 \
	--start_tensorboard=0 \
	--replay_buffer_max_size=$7 \
   	--link_delay="1ms" \
	--load_factor=0.4 \
	--sync_step=$1 \
	--max_out_buffer_size=16260 \
	--sync_ratio=0.2 \
	--signalingSim=1 \
	--nPacketsOverlay=$6 \
	--load_path=examples/$9/$4_sp_init_overlay 


array=(
0.6
0.7
0.8
0.9
1.0
1.1
1.2
1.3
1.4
)
counter=0

#For running different agents, add the following arg:
# --agent_type=sp \   #e.g., for Shortest Path agent
for j in ${array[@]}
	do 
	echo $j
	FLOAT=$(echo $j*1000 | bc)
	res2=${FLOAT/.*}
	echo $res2

	python3 main.py \
		--simTime=20 \
		--basePort=$(((4444 + $2)+(1000*$3)+($res0 * 15)+($8*1000))) \
		--train=0 \
		--seed=$2 \
		--session_name="test_new_$9_real_delay_$4_$5_real_rb_$7_sync_step_$1_variation_mat_$3_seed_$2_load_$res2" \
		--signaling_type=$5 \
		--agent_type=$4 \
		--logs_parent_folder=examples/$9/ \
		--traffic_matrix_index=$3 \
		--adjacency_matrix_path=examples/$9/adjacency_matrix.txt \
		--traffic_matrix_root_path=examples/$9/traffic_matrices/ \
		--node_coordinates_path=examples/$9/node_coordinates.txt \
		--save_models=0 \
		--start_tensorboard=0 \
		--sync_step=$1 \
		--link_delay="1ms" \
		--signalingSim=1 \
		--replay_buffer_max_size=$7 \
		--max_out_buffer_size=16260 \
		--nPacketsOverlay=$6 \
		--load_path=examples/$9/saved_models/train_new_overlay_obs_in_bytes_fixed_$9_real_delay_$4_$5_ts_0_01_seed_$2_traff_mat_$3_rb_size_$7_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_refreshRate_$6_sync_$1_loss_x11_sp_init/iteration1_episode1 \
		--load_factor=$j
	# oarsub -p "gpu='YES' and gpucapability>='5.0'" -l /nodes=1/gpunum=1,walltime=06:00:00 -d /home/ralliche/PRISMA-master/prisma/ "scripts/run_on_nef_test.sh $j 100 0 train_abilene_NN_ts_0_03_seed_100_traff_mat_0_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_sync_0.5_loss_x1_sp_init"
	counter=$((counter+1))
	echo $counter
	done
## mv -f examples/abilene/saved_models/train_abilene_NN_ts_0_03_seed_$2_traff_mat_$3_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_sync_$1_loss_x1_sp_init /data/coati/user/ralliche/examples/abilene/saved_models/
## mv -f examples/abilene/results/train_abilene_NN_ts_0_03_seed_$2_traff_mat_$3_batch_512_lr_1e-3_gamma_1_final_eps_0_01_load_40_sync_$1_loss_x1_sp_init /data/coati/user/ralliche/examples/abilene/results/
## mv -f examples/abilene/results/test_abilene_sync_step_variation_mat_$3_seed_$2_load_$res1 /data/coati/user/ralliche/examples/abilene/results/
#
## python3 main.py \
## 	--seed=1000 \
## 	--simTime=5 \
### 	--basePort=5544 \
### 	--train=0 \
## 	--agent_type=dqn_buffer \
## 	--session_name="temp" \
## 	--signaling_type=NN \
## 	--logs_parent_folder=examples/abilene/ \
## 	--traffic_matrix_root_path=examples/abilene/traffic_matrices/ \
# 	--traffic_matrix_index=0 \
# 	--adjacency_matrix_path=examples/abilene/adjacency_matrix.txt \
# 	--node_coordinates_path=examples/abilene/node_coordinates.txt \
# 	--training_step=0.01 \
# 	--batch_size=512 \
# 	--lr=0.001 \
# 	--exploration_final_eps=0.01 \
# 	--exploration_initial_eps=1.0 \
# 	--iterationNum=3000 \
# 	--gamma=1.0 \
# 	--training_trigger_type="time" \
# 	--save_models=1 \
# 	--start_tensorboard=0 \
# 	--replay_buffer_max_size=50000 \
#    	--link_delay="0ms" \
# 	--load_factor=0.01 \
# 	--sync_step=0.1 \
# 	--sync_ratio=0.2 \
# 	--signalingSim=1 \
# 	--load_path=examples/abilene/dqn_buffer_sp_init 
