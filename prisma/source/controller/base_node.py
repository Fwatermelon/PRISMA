""" This file contains abstract class for nodes. 
It is used as template to create new node. 
Each node could implement or modify the following methods:
- argument_parser: add new arguments to the global parser
- init_static_vars: init the static vars of the node and retrieve the parameters from the arg parse
- run: the main method that will be threaded
- reset: reset the node when a new episode starts
- operate: treat the given info from the env and do necessary actions
- get_action: compute the action for the given observation
- train: train the agent
- treat_lost_pkts: treat the lost packets to apply loss penalty for example
- treat_arrived_packet: treat the arrived packet to compute the reward and store the transition in the experience replay buffer for example
- communicate: communicate with other nodes (if necessary)
- log: log data (if necessary)
"""

__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2023 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


import argparse
import os
import pandas as pd
import networkx as nx
import numpy as np
from source.controller.base_logger import BaseLogger
from source.simulator.ns3gym import ns3env


class BaseNode:
    """Base class for node implementation"""

    @classmethod
    def add_arguments(
        cls, parser: (argparse.ArgumentParser)
    ) -> argparse.ArgumentParser:
        """Argument parser for this class. This method is used to add new arguments to the global parser
        Args:
            parser (argparse.ArgumentParser): parser to add arguments to
        Returns:
            argparse.ArgumentParser: parser with added arguments
        """
            ### add the arguments for the simulation
        group1 = parser.add_argument_group("Global simulation arguments")
        group1.add_argument(
            "--episode_duration", type=float, help="episode duration in seconds", default=15.0
        )
        # TODO: change basePort names
        group1.add_argument("--nb_episodes", type=int, help="Number of episodes", default=1)
        group1.add_argument(
            "--basePort", type=int, help="Starting port number", default=6555
        )
        group1.add_argument(
            "--seed", type=int, help="Random seed used for the simulation", default=100
        )
        group1.add_argument(
            "--train", type=int, help="If 1, train the model.Else, test it", default=0
        )
        group1.add_argument(
            "--max_nb_arrived_pkts",
            type=int,
            help="If < 0, stops the episode at the provided number of arrived packets",
            default=-1,
        )
        group1.add_argument(
            "--ns3_sim_path",
            type=str,
            help="Path to the ns3-gym simulator folder",
            default="../ns3-gym/",
        )
        group1.add_argument(
            "--signalingSim",
            type=int,
            help="Allows the signaling in NS3 Simulation",
            default=1,
        )
        group1.add_argument(
            "--movingAverageObsSize",
            type=int,
            help="Sets the moving average for tunnel delay values in ns3",
            default=1,
        )
        group1.add_argument(
            "--activateUnderlayTraffic",
            type=int,
            help="if 1, activate underlay traffic",
            default=0,
        )
        # TODO: set the correct default value for map overlay path
        group1.add_argument(
            "--map_overlay_path",
            type=str,
            help="Path to the map overlay file",
            default="",
        )
        group1.add_argument(
            "--pingAsObs",
            type=int,
            help="if 1, use the ping (tunnel delay) as an observation for the agent. Else, use the node output interface buffer occupancy",
            default=1,
        )
        group1.add_argument(
            "--pingPacketIntervalTime",
            type=float,
            help="Ping packet interval time P (in seconds), used when pingAsObs is 1",
            default=0.1,
        )
        ### add the arguments for the network configuration
        group2 = parser.add_argument_group("Network parameters")
        group2.add_argument(
            "--load_factor", type=float, help="scale of the traffic matrix", default=1
        )
        # TODO: set the correct default value for the traffic matrix path, the adjacency matrix path, overlay matrix path, node coordinates path
        group2.add_argument(
            "--physical_adjacency_matrix_path",
            type=str,
            help="Path to the underlay adjacency matrix",
            default="examples/abilene/physical_adjacency_matrix.txt",
        )
        group2.add_argument(
            "--overlay_adjacency_matrix_path",
            type=str,
            help="Path to the overlay adjacency matrix",
            default="examples/abilene/overlay_adjacency_matrix.txt",
        )
        group2.add_argument(
            "--tunnels_max_delays_file_name",
            type=str,
            help="Path to max observed delay per tunnel. It is to set an upper bound for the delay of the tunnels when the ping packet is lost. Used when pingAsObs is 1",
            default="examples/abilene/max_observed_values.txt",
        )
        group2.add_argument(
            "--traffic_matrix_path",
            type=str,
            help="Path to the traffic matrix folder",
            default="examples/abilene/traffic_matrices/",
        )
        group2.add_argument(
            "--node_coordinates_path",
            type=str,
            help="Path to the nodes coordinates (used for visualization)",
            default="examples/abilene/node_coordinates.txt",
        )
        group2.add_argument(
            "--max_out_buffer_size",
            type=int,
            help="Max nodes output buffer limit in bytes",
            default=16260,
        )
        group2.add_argument(
            "--link_delay", type=int, help="Network links delay in ms", default=1
        )
        group2.add_argument(
            "--packet_size", type=int, help="Size of the data packets in bytes", default=512
        )
        group2.add_argument(
            "--link_cap",
            type=int,
            help="Network links capacity in bits per seconds",
            default=500000,
        )
        group2.add_argument(
            "--auto_detect_loss",
            type=int,
            help="If 1, the network automatically detects the lost packets",
            default=0,
        )
        group2.add_argument(
            "--monitoring_type",
            choices=["active", "passive"],
            help="Type of monitoring. Active: the agents sends ping packets to monitor the network. Passive: the agents uses the reward to monitor the network",
            default="passive",
        )
        group2.add_argument(
            "--perturbations",
            type=int,
            help="If 1, the network is perturbed",
            default=0,
        )
        ### add the arguments for logging
        group3 = parser.add_argument_group("Storing session logs arguments")
        group3.add_argument(
            "--session_name",
            type=str,
            help="Name of the folder where to save the logs of the session",
            default=None,
        )
        group3.add_argument(
            "--logs_parent_folder",
            type=str,
            help="Name of the root folder where to save the logs of the sessions",
            default="examples/abilene/",
        )
        group3.add_argument(
            "--logging_timestep",
            type=int,
            help="Time delay (in real time) between each logging in seconds",
            default=5,
        )
        group3.add_argument(
            "--profile_session", type=int, help="If 1, the session is profiled (for debug)", default=0
        )
        group3.add_argument(
            "--start_tensorboard",
            type=int,
            help="if True, starts a tensorboard server to keep track of simulation progress",
            default=0,
        )
        group3.add_argument(
            "--tensorboard_port", type=int, help="Tensorboard server port (used when start_tensorboard is 1) ", default=16666
        )
        ### add the arguments for the node agent
        group4 = parser.add_argument_group("Agent type parameter")
        group4.add_argument(
            "--agent_type",
            choices=[
                "dqn_model_sharing",
                "dqn_value_sharing",
                "dqn_logit_sharing",
                "madrl_centralized",
                "shortest_path",
                "oracle_routing"
            ],
            type=str,
            help="The type of the agent. Can be based on a DQN model (dqn_model_sharing, dqn_value_sharing, dqn_logit_sharing), on the shortest path (shortest_path) or on the oracle (oracle_routing)",
            default="shortest_path",
        )
        return parser

    @classmethod
    def init_static_vars(cls, params_dict: (dict), logger: (BaseLogger)):
        """Takes the parameters of the simulation from a dict and assign it values to the static vars
        Args:
            params_dict (dict): dictionary containing the simulation parameters
            logger (BaseLogger): logger object
        """

        ### add the network topology to the params
        underlay_g = nx.DiGraph(nx.empty_graph())
        if os.path.exists(
            params_dict["node_coordinates_path"]
        ):  # load the node coordinates if they exist
            with open(params_dict["node_coordinates_path"], "r", encoding="utf-8") as f:
                pos = np.loadtxt(f)
        else:
            pos = nx.random_layout(underlay_g)
        for i, element in enumerate(pos.tolist()):
            underlay_g.add_node(i, pos=tuple(element))
        with open(
            params_dict["physical_adjacency_matrix_path"], "r", encoding="utf-8"
        ) as f:
            underlay_g = nx.from_numpy_matrix(
                np.loadtxt(f), parallel_edges=False, create_using=underlay_g
            )

        ### Add overlay topology
        if os.path.exists(params_dict["overlay_adjacency_matrix_path"]):
            with open(
                params_dict["overlay_adjacency_matrix_path"], "r", encoding="utf-8"
            ) as f:
                overlay_g = nx.from_numpy_matrix(
                    np.loadtxt(f), create_using=nx.DiGraph()
                )
        params_dict["G"] = overlay_g
        # params_dict["nn_size"] = 0
        ### fix the pingAsObs and signalingSim to 0 to disable ping packets and signaling
        params_dict["signalingSim"] = 0
        params_dict["signaling_type"] = "ideal"

        # pathlib.Path(params["logs_parent_folder"]).mkdir(parents=True, exist_ok=True)

        # network topology parameters
        cls.TOPOLOGY_PARAMS = {
            "nb_nodes": params_dict[
                "nb_nodes"
            ],  # total number of nodes in overlay topology
            "G": params_dict["G"],  # networkx graph in overlay topology
            "underlay_G": underlay_g,  # networkx graph in underlay topology
            "tunnels_max_delays_file_name": os.path.abspath(
                params_dict["tunnels_max_delays_file_name"]
            ),  # path to max observed delay per tunnel
            "auto_detect_loss": params_dict["auto_detect_loss"],  # if 1, the network automatically detects the lost packets
            "monitoring_type": params_dict["monitoring_type"],  # type of monitoring
            "monitoring_window": params_dict["movingAverageObsSize"],  # moving average for tunnel delay values
        }

        # simulation parameters
        cls.SIMULATION_PARAMS = {
            "nb_episodes": params_dict["nb_episodes"],  # number of episodes
            "episode_duration": params_dict[
                "episode_duration"
            ],  # duration of each episode
            "seed": params_dict["seed"],  # seed of the simulation
            "base_port": params_dict["basePort"],  # base port
            "ns3_sim_path": os.path.abspath(
                params_dict["ns3_sim_path"]
            ),  # path to the ns3 simulator
            "step_time": 0.1,  # step time of the simulation (used to connect to the ns3 simulator)
            "start_time": 0,  # start time of the simulation (used to connect to the ns3 simulator)
            "max_nb_arrived_pkts": params_dict[
                "max_nb_arrived_pkts"
            ],  # max number of arrived packets
            "sim_args": {
                "--simTime": params_dict["episode_duration"],
                "--testArg": 123,
            },  # arguments for the ns3 simulator (used to connect to the ns3 simulator)
            "debug": 0,  # debug level of the ns3 simulator (used to connect to the ns3 simulator)
            "logging_timestep": params_dict["logging_timestep"],  # logging interval
        }

        cls.logger = logger

        # shared attributes between nodes
        cls.shared = {
            "packets_in_network": {},  # dict containing all the packets currently in the network
            "ports_timeout": [-1 for _ in range(BaseNode.TOPOLOGY_PARAMS["nb_nodes"])],  # list of ports timeout
        }

    @classmethod
    def reset(cls):
        """Reset the nodes global attributes when a new episode starts"""
        cls.shared["packets_in_network"] = {}
        cls.logger.reset()

    def __init__(self, index: (int)):
        """Init the node
        index (int): node index
        """
        self.index = index  # node index
        self.port = BaseNode.SIMULATION_PARAMS["base_port"] + index  # port

        ### define node neighbors
        self.neighbors = list(BaseNode.TOPOLOGY_PARAMS["G"].neighbors(self.index))
        

        ### reset the node
        self.reset_node()

        ### define node attributes
        self.transition_number = 0

    def reset_node(self):
        """Reset the node when a new episode starts"""
        ### connect to the ns3 simulator
        self.env = ns3env.Ns3Env(
            stepTime=BaseNode.SIMULATION_PARAMS["step_time"],
            port=self.port,
            startSim=BaseNode.SIMULATION_PARAMS["start_time"],
            simSeed=BaseNode.SIMULATION_PARAMS["seed"],
            simArgs=BaseNode.SIMULATION_PARAMS["sim_args"],
            debug=BaseNode.SIMULATION_PARAMS["debug"],
        )
        self.transition_number = 0
        self.logger.logging["total_hops"] = 0
        

    def run(self):
        """Main method that will be threaded"""
        while True:
            obs = self.env.reset()
            pkt_id = "-1"
            while True:
                if not self.env.connected:
                    print("Not connected to the ns3 simulator")
                    break
                # print("Node", self.index, "obs", obs, "pkt_id", pkt_id)
                ### compute the action for the given observation
                action = self.get_action(obs, pkt_id)
                ### send the action to the simulator
                obs, _, flag, info = self.env.step(action)
                ### treat the given info from the env and do necessary actions
                pkt_id = self.operate(info, obs, flag)
                # print(BaseNode.SIMULATION_PARAMS["logging_timestep"] ,  BaseNode.logger.logging["last_log_time"], BaseNode.logger.logging["episode_time"] - BaseNode.logger.logging["last_log_time"] , BaseNode.logger.logging["episode_time"] , BaseNode.logger.logging["curr_time"] )
                 ### check if it is time to log and training is activated
                if BaseNode.SIMULATION_PARAMS["logging_timestep"] > 0:
                    if BaseNode.logger.logging["episode_time"] - BaseNode.logger.logging["last_log_time"] >= BaseNode.SIMULATION_PARAMS["logging_timestep"]:
                        self.log()
                        if BaseNode.logger.logging["curr_time"] <= BaseNode.SIMULATION_PARAMS["episode_duration"] - BaseNode.SIMULATION_PARAMS["logging_timestep"]:
                            self.logger.reset_moving_stats()
                        BaseNode.logger.logging["last_log_time"] = BaseNode.logger.logging["episode_time"]
                ## check if the episode is done by max number of arrived pkts
                if obs[0] == -1 and flag:  # check if the episode is done
                    print("Done by the env")
                    self.log()
                    BaseNode.logger.logging["last_log_time"] = BaseNode.logger.logging["curr_time"]
                    break
                ## check if episode is done
                if (
                    BaseNode.SIMULATION_PARAMS["max_nb_arrived_pkts"] > 0
                    and BaseNode.SIMULATION_PARAMS["max_nb_arrived_pkts"]
                    <= BaseNode.logger.logging["total_arrived_pkts"]
                ):
                    print("Done by max number of arrived pkts")
                    break
            try:
                if hasattr(self.env.ns3ZmqBridge, "send_close_command"):
                    self.env.ns3ZmqBridge.send_close_command()
            except:
                pass
            self.env.close()
            break
        return True
    
    def operate(self, info: (str), obs: (list), flag: (bool)) -> str:
        """Treat the info from the env and fill the static vars
        Args:
            info (str): info received from the ns3 simulator
            obs (list): observation
            flag (bool): if True, the packet is arrived at destination
        Returns:
            str: packet id
        """
        tokens = info.split(",")
        ### retieve simulation time
        BaseNode.logger.logging["episode_time"] = float(tokens[2].split("=")[-1])
        BaseNode.logger.logging["curr_time"] = BaseNode.logger.logging[
            "episode_time"
        ] + (
            BaseNode.logger.logging["episode_index"]
            * BaseNode.SIMULATION_PARAMS["episode_duration"]
        )
        ### retrieve packet info
        delay_time = float(tokens[0].split("=")[-1])
        pkt_size = float(tokens[1].split("=")[-1])
        pkt_id = int(tokens[3].split("=")[-1])
        pkt_type = int(tokens[4].split("=")[-1])
        if pkt_type != 0:  # control packet
            if pkt_type == 1:  # target update packet
                # pass
                raise NotImplementedError("Target update packet is not implemented")
            if pkt_type == 2:  # replay memory update packet
                return "-1"
                # raise NotImplementedError(
                #     "Replay memory update packet is not implemented"
                # )
        else:  # data packet
            ### retieve the lost packets
            lost_pkts_ids = (
                tokens[20].split("=")[-1].split(";")[:-1]
            )  # list of lost packets ids
            self.treat_lost_pkts(lost_pkts_ids)
            if obs[0] >= 0:
                ### treat the arrived packet
                self.treat_arrived_packet(pkt_id, pkt_size, delay_time, obs, flag)
            ## update stats in static variables
            self.fill_stats(tokens)
        return pkt_id

    def treat_lost_pkts(self, lost_pkts_ids: (list)):
        """Treat the lost packets
        Args:
            lost_pkts_ids (list): lost packet ids
        """
        for lost_packet_id in lost_pkts_ids:
            if (
                int(lost_packet_id) not in BaseNode.shared["packets_in_network"]
            ):  # check for a bug in the ns3 simulator
                raise ValueError("Unknown lost packet id", lost_pkts_ids, lost_packet_id, int(lost_packet_id) in BaseNode.shared["packets_in_network"].keys(), BaseNode.logger.logging["episode_time"], BaseNode.shared["packets_in_network"].keys())
            lost_packet_info = BaseNode.shared["packets_in_network"].pop(
                int(lost_packet_id)
            )
            lost_packet_info["lost_time"] = BaseNode.logger.logging["episode_time"]
            BaseNode.logger.lost_packets_per_flow[lost_packet_info["src"]][lost_packet_info["dst"]] += 1
            
            BaseNode.logger.mv_avg_loss = pd.concat(
                [
                    BaseNode.logger.mv_avg_loss,
                    pd.Series(
                        1,
                        index=[BaseNode.logger.logging["curr_time"]],
                    ),
                ]
            )
            ### Increment the loss counter
            BaseNode.logger.logging["notified_lost_pkts"] += 1

    def treat_arrived_packet(
        self,
        pkt_id: (int),
        pkt_size: (float),
        delay_time: (float),
        obs: (list),
        flag: (bool),
    ):
        """Treat the arrived packet
        Args:
            pkt_id (int): packet id
            pkt_size (float): packet size
            delay_time (float): delay taken by the packet to arrive
            obs (list): observation
            flag (bool): if True, the packet is arrived at destination
        return:
            int: the last hop of the packet. -1 if the packet is new
        """
        ### treat new arrived packet
        if pkt_id not in BaseNode.shared["packets_in_network"]:
            BaseNode.logger.logging["total_new_rcv_pkts"] += 1
            BaseNode.logger.logging["total_data_size"] += pkt_size
            BaseNode.shared["packets_in_network"][pkt_id] = {
                "src": self.index,
                "node": self.index,
                "dst": int(obs[0]),
                "hops": [self.index],
                "delays_computed": [],
                "delays_measured": 0,
                "start_time": BaseNode.logger.logging["episode_time"],
                "action_time": BaseNode.logger.logging["episode_time"],
                # "end_time": None,
                # "tag": None,
                "remaining_path": [],
            }
            BaseNode.logger.mv_injected_pkts = pd.concat(
                    [
                        BaseNode.logger.mv_injected_pkts,
                        pd.Series(
                            [pkt_size],
                            index=[BaseNode.logger.logging["curr_time"]],
                        ),
                    ]
                )
            return -1
        ### treat already arrived packet
        else:
            ### retrieve the packet info
            pkt_info = BaseNode.shared["packets_in_network"][pkt_id]
            last_hop = pkt_info["hops"][-1]
            ### compute the hop delay
            hop_time = BaseNode.logger.logging["episode_time"] - pkt_info["action_time"]
            BaseNode.logger.logging["total_rewards_with_loss"] += delay_time
            BaseNode.shared["packets_in_network"][pkt_id]["delays_computed"].append(hop_time)
            BaseNode.shared["packets_in_network"][pkt_id]["delays_measured"]= delay_time
            BaseNode.shared["packets_in_network"][pkt_id]["hops"].append(self.index)
            BaseNode.shared["packets_in_network"][pkt_id]["node"] = self.index
            if not flag:  # handle transit packets
                ## update the packet info in the dict
                BaseNode.shared["packets_in_network"][pkt_id][
                    "action_time"
                ] = BaseNode.logger.logging["episode_time"]
            else:  # handle packet that arrived at destination
                BaseNode.logger.logging["total_arrived_pkts"] += 1
                nb_hops = len(BaseNode.shared["packets_in_network"][pkt_id]["hops"]) - 1
                BaseNode.logger.logging["total_hops"] += nb_hops
                BaseNode.logger.logging["total_e2e_delay_measured"] += delay_time
                BaseNode.logger.logging["total_e2e_delay_computed"] += BaseNode.logger.logging["episode_time"] - pkt_info["start_time"]
                packet_info = BaseNode.shared[
                    "packets_in_network"
                ].pop(pkt_id)
                BaseNode.logger.e2e_delay_per_flow[packet_info["src"]][packet_info["dst"]].append(delay_time)
                BaseNode.logger.nb_hops_per_flow[packet_info["src"]][packet_info["dst"]].append(len(packet_info["hops"])-1)
                BaseNode.logger.paths_per_flow[packet_info["src"]][packet_info["dst"]].add(str(packet_info["hops"]))
                
                BaseNode.logger.mv_avg_e2e_delay = pd.concat(
                    [
                        BaseNode.logger.mv_avg_e2e_delay,
                        pd.Series(
                            [delay_time],
                            index=[BaseNode.logger.logging["curr_time"]],
                        ),
                    ]
                )
                BaseNode.logger.mv_nb_hops = pd.concat(
                    [
                        BaseNode.logger.mv_nb_hops,
                        pd.Series(
                            [len(packet_info["hops"]) - 1],
                            index=[BaseNode.logger.logging["curr_time"]],
                        ),
                    ]
                )
                
            return last_hop

    def fill_stats(self, tokens: (list)):
        """Fill the static vars with the given tokens
        Args:
            tokens (list): list of tokens
        """
        # print("Filling stats with tokens", tokens)
        BaseNode.logger.logging["sim_avg_e2e_delay"] = float(tokens[5].split("=")[-1])
        BaseNode.logger.logging["sim_cost"] = float(tokens[6].split("=")[-1])
        BaseNode.logger.logging["sim_global_avg_e2e_delay"] = float(
            tokens[7].split("=")[-1]
        )
        BaseNode.logger.logging["sim_global_cost"] = float(tokens[8].split("=")[-1])
        BaseNode.logger.logging["sim_dropped_packets"] = float(tokens[9].split("=")[-1])
        BaseNode.logger.logging["sim_rejected_packets"] = float(
            tokens[10].split("=")[-1]
        )
        BaseNode.logger.logging["sim_delivered_packets"] = float(
            tokens[11].split("=")[-1]
        )
        BaseNode.logger.logging["sim_injected_packets"] = float(
            tokens[12].split("=")[-1]
        )
        BaseNode.logger.logging["sim_buffered_packets"] = float(
            tokens[13].split("=")[-1]
        )
        BaseNode.logger.logging["sim_global_dropped_packets"] = (
            float(tokens[14].split("=")[-1])
            + BaseNode.logger.logging["sim_dropped_packets"]
        )
        BaseNode.logger.logging["sim_global_rejected_packets"] = float(
            tokens[15].split("=")[-1]
        )
        BaseNode.logger.logging["sim_global_delivered_packets"] = (
            float(tokens[16].split("=")[-1])
            + BaseNode.logger.logging["sim_delivered_packets"]
        )
        BaseNode.logger.logging["sim_global_injected_packets"] = (
            float(tokens[17].split("=")[-1])
            + BaseNode.logger.logging["sim_injected_packets"]
        )
        BaseNode.logger.logging["sim_global_buffered_packets"] = (
            float(tokens[18].split("=")[-1])
            + BaseNode.logger.logging["sim_buffered_packets"]
        )
        BaseNode.logger.logging["sim_signaling_overhead"] = float(
            tokens[19].split("=")[-1]
        )
        if BaseNode.logger.logging["sim_global_delivered_packets"] > 0:
            BaseNode.logger.logging["sim_global_avg_e2e_delay"] = (
                (
                    BaseNode.logger.logging["sim_global_avg_e2e_delay"]
                    * float(tokens[16].split("=")[-1])
                )
                + (
                    BaseNode.logger.logging["sim_avg_e2e_delay"]
                    * BaseNode.logger.logging["sim_delivered_packets"]
                )
            ) / (BaseNode.logger.logging["sim_global_delivered_packets"])

        if (
            BaseNode.logger.logging["sim_global_delivered_packets"]
            + BaseNode.logger.logging["sim_global_dropped_packets"]
            > 0
        ):
            BaseNode.logger.logging["sim_global_cost"] = (
                (
                    BaseNode.logger.logging["sim_global_cost"]
                    * (
                        float(tokens[16].split("=")[-1])
                        + float(tokens[14].split("=")[-1])
                    )
                )
                + (
                    BaseNode.logger.logging["sim_cost"]
                    * (
                        BaseNode.logger.logging["sim_dropped_packets"]
                        + BaseNode.logger.logging["sim_delivered_packets"]
                    )
                )
            ) / (
                BaseNode.logger.logging["sim_global_dropped_packets"]
                + BaseNode.logger.logging["sim_global_delivered_packets"]
            )

    def log(self):
        """ Log data into tensorboard"""
        ### log the data
        BaseNode.logger.log_stats()

            
    def get_action(self, obs: (list), pkt_id: (str)) -> int:
        """Forward the packet
        Args:
            obs (list): observation
            pkt_id (str): packet id
        Returns:
            int: action
        """
        if obs[0] < 0 or pkt_id == "-1":
            return 0
        ### treat the case when an action is not required
        action = 0
        ## Increment the simulation and episode counters
        BaseNode.logger.logging["total_nb_iterations"] += 1
        BaseNode.logger.logging["nodes_nb_iterations"][self.index] += 1
        return action

    def close(self):
        """Close the node"""
        ## close the zmq bridge
        # try:
        #     if hasattr(self.env.ns3ZmqBridge, "send_close_command"):
        #         self.env.ns3ZmqBridge.send_close_command()
        # except:
        #     pass
        # self.env.close()
        pass
        # TODO : save test results
        #     ## write the results for the test session
        # if params["train"] == 0:
        #     print("Writing test results in ", params["logs_folder"] + "/test_results")
        #     stats_writer_test(params["logs_folder"] + "/test_results", Agent)

        #     # move the logs folder to the session folder and delete the old one if it exists

        #     logs_folder_path = os.path.join(
        #         params["logs_parent_folder"], params["session_name"]
        #     )
        #     if os.path.exists(logs_folder_path):
        #         shutil.rmtree(logs_folder_path)
        #     os.rename(params["logs_folder"], logs_folder_path)
        #     print(f"Logs folder moved to {logs_folder_path}/")
