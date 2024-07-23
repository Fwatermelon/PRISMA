""" This file contains the class for the shortest path agent. It is used to run the shortest path algorithm on the overlay topology.
"""

__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2023 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


import argparse
import json
import os
import numpy as np
import networkx as nx
from source.controller.base_node import BaseNode
from source.controller.base_logger import BaseLogger


class ShortestPathAgent(BaseNode):
    """class for the shortest path agent. It is used to run the shortest path algorithm on the overlay topology."""

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
        # add the option to have k shortest paths (ECMP)
        group = parser.add_argument_group("Shortest Path")
        group.add_argument("--K", type=int, help="Number of shortest paths, if > 1, use equally cost load balancing (ECMP)", default=1)
        return parser

    @classmethod
    def init_static_vars(cls, params_dict: (dict), logger: (BaseLogger)):
        """Takes the parameters of the simulation from a dict and assign it values to the static vars"""
        BaseNode.init_static_vars(params_dict, logger)
        cls.K = params_dict["K"]
        
        
    def __init__(self, index):
        """Init the node
        index (int): node index
        """
        super().__init__(index)

        # load balancing approach
        ### computing the routing table
        self.routing_table = []
        for dst in range(ShortestPathAgent.TOPOLOGY_PARAMS["nb_nodes"]):
            self.routing_table.append([])
            if dst == self.index:
                self.routing_table[dst].append([-1])
            else:
                paths_iterator = nx.shortest_simple_paths(ShortestPathAgent.TOPOLOGY_PARAMS["G"], self.index, dst)
                for _ in range(ShortestPathAgent.K):
                    try:
                        path = next(paths_iterator)
                        self.routing_table[dst].append(path[1:])
                    except StopIteration:
                        pass
        print("Shortest path agent initialized", self.index, self.routing_table)

    def get_action(self, obs: (list), pkt_id: (str)) -> int:
        """Forward the packet following the shortest path. if K > 1, use equally cost load balancing (ECMP)
        Args:
            obs (list): observation
            pkt_id (str): packet id
        Returns:
            int: action
        """
        ### treat the case when an action is not required
        if obs[0] < 0 or pkt_id == "-1" or obs[0] == self.index:
            return 0
        ### check if the packets is not new (already in the network)
        if pkt_id in ShortestPathAgent.shared["packets_in_network"]:
            if "remaining_path" in ShortestPathAgent.shared["packets_in_network"][pkt_id]:
                if len(ShortestPathAgent.shared["packets_in_network"][pkt_id]["remaining_path"]) > 0:
                    action = self.neighbors.index(ShortestPathAgent.shared["packets_in_network"][pkt_id]["remaining_path"].pop(0))
                else:
                    ### get the path by random choice
                    ShortestPathAgent.shared["packets_in_network"][pkt_id]["remaining_path"] = list(self.routing_table[int(obs[0])][np.random.randint(0, len(self.routing_table[int(obs[0])]))])
                    # print("Packet", pkt_id, "is in the shared memory but has no remaining path. Assigning a new path", ShortestPathAgent.shared["packets_in_network"][pkt_id]["remaining_path"], "chosen randomly from the routing table", self.routing_table[int(obs[0])])
                    action = self.neighbors.index(ShortestPathAgent.shared["packets_in_network"][pkt_id]["remaining_path"].pop(0))
            else:
                raise ValueError("Packet", pkt_id, "is not in the shared memory")
        else:
            raise ValueError("Packet", pkt_id, "is not in the shared memory")
        ### Increment the simulation and episode counters
        BaseNode.logger.logging["total_nb_iterations"] += 1
        BaseNode.logger.logging["nodes_nb_iterations"][self.index] += 1
        return action



class OracleAgent(BaseNode):
    """Class node for the oracle agent. It is used to run the oracle routing algorithm on the overlay topology."""

    @classmethod
    def add_arguments(cls, parser: (argparse.ArgumentParser)) -> argparse.ArgumentParser:
        """Argument parser for this class. This method is used to add new arguments to the global parser
        Args:
            parser (argparse.ArgumentParser): parser to add arguments to
        Returns:
            argparse.ArgumentParser: parser with added arguments
        """
        gr = parser.add_argument_group("Oracle")
        gr.add_argument("--optimal_solution_path", type=str, help="Path to the optimal solution file", default=".")
        gr.add_argument("--opt_rejected_path", type=str, help="Path to the optimal rejected flows file", default=".")
        return parser

    @classmethod
    def init_static_vars(cls, params_dict: (dict), logger: (BaseLogger)):
        """Takes the parameters of the simulation from a dict and assign it values to the static vars"""
        BaseNode.init_static_vars(params_dict, logger)
        ### read the optimal solution file
        # TODO: check if the optimal solution matches the topology size and the traffic matrix
        cls.optimal_solution = json.load(open(params_dict["optimal_solution_path"], "r", encoding="utf-8"))
        cls.optimal_rejected = cls.optimal_solution["rejected_flows"]
        ### check if the rejected flows file is not given
        if not os.path.isfile(params_dict["opt_rejected_path"]):
            params_dict["opt_rejected_path"] = params_dict["optimal_solution_path"].replace(".json", "_rejected.txt")
            ### save the reject flows in a file
            np.savetxt(params_dict["opt_rejected_path"], cls.optimal_rejected, fmt="%f")
        ### fix the paths
        params_dict["optimal_solution_path"] = os.path.abspath(params_dict["optimal_solution_path"])
        params_dict["opt_rejected_path"] = os.path.abspath(params_dict["opt_rejected_path"])
        print("Oracle agent initialized", params_dict["optimal_solution_path"], params_dict["opt_rejected_path"])


    def __init__(self, index):
        """Init the node
        index (int): node index
        """
        super().__init__(index)
        ### get the indices of the links related to the node
        self.links = np.array(list(OracleAgent.optimal_solution["overlay_links"]))
        self.link_indices = np.where(self.links[:, 0] == self.index)[0]
        self.routing_probs = np.array(OracleAgent.optimal_solution["routing"][self.index])

    def get_action(self, obs: (list), pkt_id: (str)) -> int:
        """Forward the packet following the oracle routing algorithm.
        For each incoming packet decide on the path to take based on the optimal solution.
        Since the optimal solution is computed using link repartition for each source destination pair, the path is constructed hop by hop.
        Args:
            obs (list): observation
            pkt_id (str): packet id
        Returns:
            int: action
        """
        ### treat the case when an action is not required
        if obs[0] < 0 or pkt_id == "-1" or obs[0] == self.index:
            return 0
        probs = np.array(self.routing_probs[obs[0]])[self.link_indices]
        ### normalize the probabilities
        probs = probs / np.sum(probs)
        ### get the action by random choice given the probabilities
        action = self.neighbors.index(np.random.choice(self.links[:, 1][self.link_indices], p=probs))
        ### Increment the simulation and episode counters
        BaseNode.logger.logging["total_nb_iterations"] += 1
        BaseNode.logger.logging["nodes_nb_iterations"][self.index] += 1
        return action