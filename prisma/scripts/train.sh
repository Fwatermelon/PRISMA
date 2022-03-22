#!/bin/bash  
 
## Move to main folder
cd ..

## Copy prisma into ns-3 folder
rsync -r --exclude-from=../.gitignore ../prisma ../ns3-gym/scratch/

## configure ns3
cd ../ns3-gym
mv scratch/prisma/ns3/* scratch/prisma/.

./waf -d optimized configure
sleep 3
cd ../prisma

## Training DQN

#For running different agents, add the following arg:
# --agent_type=sp \   #e.g., for Shortest Path agent

python3 main.py --simTime=60 \
	--basePort=6555 \
	--train=1 \
	--session_name="test"\
	--logs_parent_folder=examples/geant/ \
	--traffic_matrix_path=examples/geant/traffic_matrices/node_intensity_normalized.txt \
	--adjacency_matrix_path=examples/geant/adjacency_matrix.txt \
	--node_coordinates_path=examples/geant/node_coordinates.txt \
	--training_step=0.007 \
	--batch_size=512 \
	--lr=1e-4 \
	--exploration_final_eps=0.1 \
	--exploration_initial_eps=1.0 \
	--iterationNum=3000 \
	--gamma=1 \
	--training_trigger_type="time" \
	--save_models=1 \
	--start_tensorboard=0 \
	--load_factor=0.5
sleep 5


rm -r ../ns3-gym/scratch/prisma
