""" Argument parser for the simulation """
__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

import argparse
import os
from source.controller.base_node import BaseNode
from source.controller.agents.base_dqn_routing import BaseDqnRoutingAgent
# from source.controller.agents.dqn_routing_value_sharing import ValueSharingAgent
from source.controller.agents.dqn_routing_logit_sharing import LogitSharingAgent
from source.controller.agents.dqn_routing_model_sharing import ModelSharingAgent
from source.controller.agents.classical_routing_methods import ShortestPathAgent, OracleAgent
from source.controller.agents.dqn_routing_centralized import CentralizedMADRLAgent

def parse_arguments():
    """Retrieve and parse argument from the commandline"""
    ## Setup the argparser
    description_txt = """PRISMA : Packet Routing Simulator for Multi-Agent Reinforcement Learning.
    PRISMA is a network simulation playground for developing and testing Multi-Agent Reinforcement Learning (MARL) solutions for dynamic packet routing (DPR) in overlay networks. 
    This framework is based on the OpenAI Gym toolkit and the Ns3 simulator.
    """
    # TODO: change the example
    epilog_txt = """Example:
                    python3 main.py --agent_type=shortest_path --nb_episodes=1 --episode_duration=60 --seed=100 --basePort=6555 --train=0 --max_nb_arrived_pkts=-1 --ns3_sim_path=../ns3-gym/ --signalingSim=0 --movingAverageObsSize=5 --activateUnderlayTraffic=0 --map_overlay_path=examples/abilene/overlay_adjacency_matrix.txt --pingAsObs=1 --pingPacketIntervalTime=0.2 --d_t_send_all_destinations=0 --load_factor=1 --physical_adjacency_matrix_path=examples/abilene/physical_adjacency_matrix.txt --overlay_adjacency_matrix_path=examples/abilene/overlay_adjacency_matrix.txt --tunnels_max_delays_file_name=examples/abilene/max_observed_values.txt --traffic_matrix_path=examples/abilene/traffic_matrices/ --node_coordinates_path=examples/abilene/node_coordinates.txt --max_out_buffer_size=16260 --link_delay=1 --packet_size=512 --link_cap=500000 --session_name=abilene --logs_parent_folder=examples/abilene/ --profile_session=0 --start_tensorboard=0 --tensorboard_port=16666
                    """

    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="python3 %(prog)s [options]",
        description=description_txt,
        epilog=epilog_txt,
        allow_abbrev=False,
    )
    ### add the arguments for the base node
    BaseNode.add_arguments(parser)
    ### add the arguments for classical agents
    ShortestPathAgent.add_arguments(parser)
    OracleAgent.add_arguments(parser)
    ### add the arguments for the DQN agents
    BaseDqnRoutingAgent.add_arguments(parser)
    ModelSharingAgent.add_arguments(parser)
    LogitSharingAgent.add_arguments(parser)
    CentralizedMADRLAgent.add_arguments(parser)
    # ValueSharingAgent.add_arguments(parser)
    # parser.print_help()
    ## get the params dict
    params = vars(parser.parse_args())
    params["nb_nodes"] = len(open(params["overlay_adjacency_matrix_path"], encoding="utf-8").readlines())
    print("params = ", params)
    ### disable the signaling if train is 0
    if params["train"] == 0:
        params["signalingSim"] = 0
    ### resolve the paths
    params["logs_parent_folder"] = os.path.abspath(params["logs_parent_folder"])
    params["physical_adjacency_matrix_path"] = os.path.abspath(
        params["physical_adjacency_matrix_path"]
    )
    params["overlay_adjacency_matrix_path"] = os.path.abspath(
        params["overlay_adjacency_matrix_path"]
    )
    params["tunnels_max_delays_file_name"] = os.path.abspath(
        params["tunnels_max_delays_file_name"]
    )
    # TODO : check the given paths and verify the size of the matrices
    params["traffic_matrix_path"] = os.path.abspath(params["traffic_matrix_path"])
    params["node_coordinates_path"] = os.path.abspath(params["node_coordinates_path"])
    params["map_overlay_path"] = os.path.abspath(params["map_overlay_path"])
    params["ns3_sim_path"] = os.path.abspath(params["ns3_sim_path"])
    
    return params
