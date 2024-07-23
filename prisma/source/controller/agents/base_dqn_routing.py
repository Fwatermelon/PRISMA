""" This file contains abstract class for nodes. 
It is used as template to create new node. 
Each node should implement the following methods:
- run: the main method that will be threaded
- reset: reset the node when a new episode starts
- operate: treat the given info from the env and do necessary actions
- get_action: compute the action for the given observation
- train: train the agent
- treat_lost_pkts: treat the lost packets to apply loss penalty for example
- treat_arrived_packet: treat the arrived packet to compute the reward and store the transition in the experience replay buffer for example
- treat_memory_update_pkt: treat the replay memory update packet (if necessary)
- treat_target_update_pkt: treat the target update packet (if necessary)
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
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from source.controller.agents.learner import LinearSchedule, load_model, DQN_AGENT
from source.controller.agents.neural_nets_architectures import DQN_buffer_model
from source.simulator.ns3gym import ns3env
from source.controller.base_node import BaseNode
from source.controller.agents.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class BaseDqnRoutingAgent(BaseNode):
    """Base class for DQN routing"""

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
        group0 = parser.add_argument_group("RL arguments")
        group0.add_argument(
            "--lr", type=float, help="Learning rate (used when training)", default=1e-4
        )
        group0.add_argument(
            "--prioritizedReplayBuffer",
            type=int,
            help="if true, use prioritized replay buffer using the gradient step as weights (used when training)",
            default=0,
        )

        group0.add_argument(
            "--batch_size",
            type=int,
            help="Size of a batch (used when training)",
            default=512,
        )
        group0.add_argument(
            "--gamma",
            type=float,
            help="Gamma ratio for RL (used when training)",
            default=1,
        )
        group0.add_argument(
            "--iterationNum",
            type=int,
            help="Max iteration number for exploration (used when training)",
            default=5000,
        )
        group0.add_argument(
            "--exploration_initial_eps",
            type=float,
            help="Exploration intial value (used when training)",
            default=1.0,
        )
        group0.add_argument(
            "--exploration_final_eps",
            type=float,
            help="Exploration final value (used when training)",
            default=0.1,
        )
        group0.add_argument(
            "--reset_exploration",
            type=int,
            help="Reset the exploration after each episode (used when training)",
            default=0,
        )
        group0.add_argument(
            "--load_path",
            type=str,
            help="Path to DQN models, if not None, loads the models from the given files",
            default=None,
        )
        group0.add_argument(
            "--save_models",
            type=int,
            help="if True, store the models at the end of the training",
            default=0,
        )
        group0.add_argument(
            "--saved_models_path",
            type=str,
            help="Path to saved models (NNs)",
            default=None,
        )
        group0.add_argument(
            "--snapshot_interval",
            type=int,
            help="Number of seconds between each snapshot of the models. If 0, desactivate snapshot",
            default=0,
        )
        group0.add_argument(
            "--training_step",
            type=float,
            help="Number of steps or seconds to train (used when training)",
            default=0.1,
        )
        group0.add_argument(
            "--replay_buffer_max_size",
            type=int,
            help="Max size of the replay buffers (used when training)",
            default=50000,
        )
        ### add arguments for reward shaping
        group1 = parser.add_argument_group("Reward shaping arguments")
        group1.add_argument(
            "--loss_penalty_type",
            type=str,
            choices=["None", "fixed", "guided"],
            help="Define how to consider the lost packets into the reward. If None, no loss penalty. If fixed, use a fixed loss pen. If guided, use a guided loss mechanism based on RCPO",
            default="fixed",
        )
        group1.add_argument(
            "--fixed_loss_penalty",
            type=float,
            help="Set the fixed loss penalty (used when training and loss_penalty_type is fixed)",
            default=3,
        )
        # group1.add_argument(
        #     "--lamda_training_start_time",
        #     type=float,
        #     help="Number of seconds to wait before starting the training (used when training and loss_penalty_type is guided)",
        #     default=60,
        # )
        group1.add_argument(
            "--lambda_lr",
            type=float,
            help="Learning rate for the lambda parameter (used when training and loss_penalty_type is guided)",
            default=1e-7,
        )
        group1.add_argument(
            "--lambda_train_step",
            type=float,
            help="Number of seconds to train (used when training)",
            default=-1,
        )
        group1.add_argument(
            "--alpha",
            type=float,
            help="Alpha parameter for constraint violation (used when training and loss_penalty_type is guided)",
            default=1,
        )
    
        # group1.add_argument(
        #     "--rcpo_consider_loss",
        #     type=int,
        #     help="Whether to consider lost packets as a constraint violation (used when training)",
        #     default=1,
        # )
        # group1.add_argument(
        #     "--rcpo_use_loss_pkts",
        #     type=int,
        #     help="Whether to use the number of lost packets as a constraint (used when training)",
        #     default=0,
        # )
        ### add arguments for experience relevancy
        group2 = parser.add_argument_group("Experience relevancy arguments")
        group2.add_argument(
            "--gap_threshold",
            type=float,
            help="Error threshold between the node and the neighors estimations in order to trigger a replay memory update packet (used when training for model sharing)",
            default=0,
        )

        return parser

    @classmethod
    def init_static_vars(cls, params_dict, logger):
        """Takes the parameters of the simulation from a dict and assign it values to the static vars
        Args:
            params_dict (dict): dictionary containing the simulation parameters
            logger (BaseLogger): logger object
        """
        BaseNode.init_static_vars(params_dict, logger)
        # TODO: fix the logs folder
        # if params["train"] == 1:
        #     pathlib.Path("/tmp/logs").mkdir(parents=True, exist_ok=True)
        #     params["logs_folder"] = "/tmp/logs/" + params["session_name"]
        # else:
        ### fix the paths to absolute paths for saving and loading the models
        if params_dict["saved_models_path"]:
            params_dict["saved_models_path"] = os.path.abspath(
                params_dict["saved_models_path"]
            )
        else:
            params_dict["saved_models_path"] = (
                params_dict["logs_parent_folder"] + "/saved_models/"
            )
        ## TODO: fix ping as obs when we are in single domain and disable it when we are in multi domain
        
        ### set the signalling to 0 if train is 0
        if params_dict["train"] == 0:
            params_dict["signalingSim"] = 0
            params_dict["signaling_type"] = "ideal"
        else:
            params_dict["signalingSim"] = 1

        ### add learning parameters for RL
        cls.RL_PARAMS = {
            "train": params_dict[
                "train"
            ],  # if true, toggle training mode. If false, toggle testing mode
            "lr": params_dict["lr"],  # learning rate
            "batch_size": params_dict["batch_size"],  # batch size
            "gamma": params_dict["gamma"],  # discount factor
            "exploration_initial_eps": params_dict[
                "exploration_initial_eps"
            ],  # initial exploration rate
            "exploration_final_eps": params_dict[
                "exploration_final_eps"
            ],  # final exploration rate
            "iterationNum": params_dict["iterationNum"],  # max iteration number
            "reset_exploration": params_dict[
                "reset_exploration"
            ],  # if true, reset the exploration after each episode
            "replay_buffer_max_size": params_dict[
                "replay_buffer_max_size"
            ],  # max size of the experience replay memory buffer
            "prioritizedReplayBuffer": params_dict[
                "prioritizedReplayBuffer"
            ],  # if true, use prioritized experience replay
            "load_path": params_dict[
                "load_path"
            ],  # path to load the NN weights for the agent
            "save_models": params_dict[
                "save_models"
            ],  # if true, save the models at the end of the training
            "saved_models_path": params_dict[
                "saved_models_path"
            ],  # path to save the models
            "snapshot_interval": params_dict[
                "snapshot_interval"
            ],  # number of seconds between each snapshot of the models. If 0, desactivate snapshot
            "training_step": params_dict[
                "training_step"
            ],  # number of steps or seconds to train, noted T in the paper
            "loss_penalty_type": params_dict[
                "loss_penalty_type"
            ],  # define how to consider the lost packets into the reward. If None, no loss penalty. If fixed, use a fixed loss pen. If guided, use a guided loss mechanism based on RCPO
            "fixed_loss_penalty": params_dict[
                "fixed_loss_penalty"
            ],  # set the fixed loss penalty
            "agent_type": params_dict["agent_type"],  # agent type
                
        }
        cls.EXPERIENCE_RELEVANCY_PARAMS = {
            "gap_threshold": params_dict["gap_threshold"]
        }

        ### add parameters for the guided reward
        cls.GUIDED_REWARD_PARAMS = {
            # "lamda_training_start_time": params_dict["lamda_training_start_time"],
            "lambda_lr": params_dict["lambda_lr"],
            "lambda_train_step": params_dict["lambda_train_step"], 
            "alpha": 1, # alpha parameter for constraint violation
        }
        
        ### shared attributes between nodes
        cls.shared["DRL_agents"] = {
                i: None for i in range(params_dict["nb_nodes"])
            }  # dict containing all the agents
        cls.shared["replay_buffers"] = [
                PrioritizedReplayBuffer(
                    params_dict["replay_buffer_max_size"],
                    1,
                    len(list(params_dict["G"].neighbors(n))),
                    n,
                )
                for n in range(params_dict["nb_nodes"])
            ] if params_dict["prioritizedReplayBuffer"] else [
                ReplayBuffer(params_dict["replay_buffer_max_size"])
                for _ in range(params_dict["nb_nodes"])
            ]  # replay buffer
        cls.shared["nn_locks"]= [
                threading.Lock() for _ in range(params_dict["nb_nodes"])
            ]  # lock used to avoid concurrent access to the neural network
        cls.shared["last_training_time"] = 0  # last time the agents were trained
        cls.shared["temp_replay_buffers"] = [ {} for _ in range(params_dict["nb_nodes"])] # temporary replay buffer used to store the transitions before adding them to the replay buffer
        cls.shared["last_log_time"]= 0 # last time the data was logged
        cls.shared["last_lambda_training_time"] = 0 # last time the lambda coefficients were trained
        cls.shared["nb_loss_pkts"] = [[0 for _ in range(len(list(params_dict["G"].neighbors(n))))] for n in range(params_dict["nb_nodes"])]  # lost pkt per node per port
        ### import max delays from the file
        if params_dict["tunnels_max_delays_file_name"]:
            cls.shared["max_delays"] = np.loadtxt(params_dict["tunnels_max_delays_file_name"])
            
        
        cls.shared["lambda_coefs"] =  [[cls.shared["max_delays"][n][j] for j in range(len(list(cls.TOPOLOGY_PARAMS["G"].neighbors(n))))] for n in range(cls.TOPOLOGY_PARAMS["nb_nodes"])]
        params_dict["opt_rejected_path"] = "/app/prisma/examples/overlay_full_mesh_10n_geant/optimal_solution/0_norm_matrix_uniform/60_ut_minCostMCF_rejected.txt"

        ### init the agents
        for index in range(params_dict["nb_nodes"]):
            nb_neighbors = len(list(params_dict["G"].neighbors(index)))
            observation_shape = (nb_neighbors + 1,)
            cls.shared["DRL_agents"][index] = DQN_AGENT(
                q_func=DQN_buffer_model,
                observation_shape=observation_shape,
                num_actions=nb_neighbors,
                nb_nodes=params_dict["nb_nodes"],
                input_size_splits=[
                    1,
                    nb_neighbors,
                ],
                lr=params_dict["lr"],
                gamma=params_dict["gamma"],
                neighbors_degrees=[
                    len(list(params_dict["G"].neighbors(x))) for x in list(params_dict["G"].neighbors(index))
                ],
                d_t_max_time=params_dict["target_update_period"],
                d_q_func=DQN_buffer_model,
            )
            if params_dict["load_path"]:
                if os.path.exists(os.path.abspath(params_dict["load_path"])):
                    q_vars = load_model(os.path.abspath(params_dict["load_path"]), index)[index].trainable_variables
                    target_q_vars = cls.shared["DRL_agents"][index].q_network.trainable_variables
                    for var, var_target in zip(q_vars, target_q_vars):
                        var_target.assign(var)
                else:
                    raise ValueError("The load path does not exist")

    @classmethod
    def reset(cls):
        """Reset the nodes global attributes when a new episode starts"""
        BaseNode.reset()
        cls.shared["last_training_time"] = 0
        cls.shared["last_log_time"] = 0
        cls.shared["last_lambda_training_time"] = 0

    

    def __init__(self, index):
        """Init the node
        index (int): node index
        """
        super().__init__(index)
        self.update_eps = 0 # used for the exploration


    def reset_node(self):
        """Reset the node when a new episode starts"""
        super().reset_node()
        ### reset exploration
        if BaseDqnRoutingAgent.RL_PARAMS["reset_exploration"] or BaseDqnRoutingAgent.logger.logging["episode_index"] ==0:
            if BaseDqnRoutingAgent.RL_PARAMS["exploration_final_eps"] > BaseDqnRoutingAgent.RL_PARAMS["exploration_initial_eps"]:
                raise ValueError("exploration_final_eps must be smaller than or equal to exploration_initial_eps")
            if BaseDqnRoutingAgent.RL_PARAMS["iterationNum"] <= 0:
                raise ValueError("iterationNum must be greater than 0")
            if BaseDqnRoutingAgent.RL_PARAMS["train"]:
                self.exploration = LinearSchedule(
                    schedule_timesteps=BaseDqnRoutingAgent.RL_PARAMS["iterationNum"], initial_p=BaseDqnRoutingAgent.RL_PARAMS["exploration_initial_eps"], final_p=BaseDqnRoutingAgent.RL_PARAMS["exploration_final_eps"]
                )
            else:
                self.exploration = LinearSchedule(
                    schedule_timesteps=1, initial_p=0, final_p=0
                )
        ### reset the temporary replay buffer
        BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index] = {}
        BaseDqnRoutingAgent.shared["packets_in_network"] = {}
        BaseDqnRoutingAgent.shared["lost_detected"] = []
        BaseDqnRoutingAgent.logger.logging["replay_memory_update_overhead_size"] = 0
        BaseDqnRoutingAgent.logger.logging["nb_memory_update_pkts"] = 0
        BaseDqnRoutingAgent.logger.logging["target_update_overhead_size"] = 0
        BaseDqnRoutingAgent.logger.logging["nb_target_update_pkts"] = 0
        BaseDqnRoutingAgent.logger.logging["lost_pkts_detected"] = 0
        BaseDqnRoutingAgent.logger.logging["notified_lost_pkts"] = 0
        
        # if monitoring is passive, reset the moving average delays
        if BaseDqnRoutingAgent.TOPOLOGY_PARAMS["monitoring_type"] == "passive":
            self.moving_avg_delays = [[0 for _ in range(BaseDqnRoutingAgent.TOPOLOGY_PARAMS["monitoring_window"])] for _ in range(len(list(BaseDqnRoutingAgent.TOPOLOGY_PARAMS["G"].neighbors(self.index))))]

        
        
    def get_action(self, obs: (list), pkt_id: (str), last_hop: (int)) -> int:
        """Forward the packet using the neural network.
        Args:
            obs (list): observation
            pkt_id (str): packet id
            last_hop (int): last hop
        Returns:
            int: action. if action= 1000 + output interface, it means that the node should not send the signalling packet after data packet reception.
        """
        ### treat the case when an action is not required
        if obs[0] < 0 or pkt_id is None or len(obs) == 1:
            return 0
        
        ### if the packet is at the destination
        if obs[0] == self.index:
            if BaseDqnRoutingAgent.RL_PARAMS["train"] and BaseDqnRoutingAgent.EXPERIENCE_RELEVANCY_PARAMS["gap_threshold"] > 0:
                ### if dynamic send signalling is activated, check if the node should send a signalling packet
                gap = self.compute_gap(pkt_id, [], last_hop)
                if gap>= 0 and gap <= BaseDqnRoutingAgent.EXPERIENCE_RELEVANCY_PARAMS["gap_threshold"]:
                    ## remove the packet from the temporary replay buffer      
                    BaseDqnRoutingAgent.shared["temp_replay_buffers"][last_hop][pkt_id].pop(-1)
                    if len(BaseDqnRoutingAgent.shared["temp_replay_buffers"][last_hop][pkt_id]) == 0:
                        BaseDqnRoutingAgent.shared["temp_replay_buffers"][last_hop].pop(pkt_id)
                    return 1000
            return 0
        
        ### train treatment for transition packet
        if BaseDqnRoutingAgent.RL_PARAMS["train"]:
            ### update epsilon for the exploration
            self.update_eps = tf.constant(
                self.exploration.value(BaseDqnRoutingAgent.logger.logging["nodes_nb_iterations"][self.index])
            )
            BaseDqnRoutingAgent.logger.logging["epsilon"][self.index] = self.update_eps.numpy().item()

        ## get the action
        BaseDqnRoutingAgent.shared["nn_locks"][self.index].acquire()
        action, q_val = BaseDqnRoutingAgent.shared["DRL_agents"][self.index].step(np.array([obs]), update_eps=self.update_eps, stochastic=BaseDqnRoutingAgent.RL_PARAMS["train"])
        BaseDqnRoutingAgent.shared["nn_locks"][self.index].release()
        action = action.numpy().item()
        q_val = q_val.numpy().tolist()[0]
        # print("take action", action, "with q value", q_val, "for obs", obs, "and epsilon", self.update_eps.numpy().item(), "and pkt id", pkt_id, "at node", self.index, "time", BaseDqnRoutingAgent.logger.logging["curr_time"])
        if BaseDqnRoutingAgent.RL_PARAMS["train"]:
            send_signalling = 1000 # do not send signalling packet by default
            if last_hop >= 0 and BaseDqnRoutingAgent.EXPERIENCE_RELEVANCY_PARAMS["gap_threshold"] > 0:
                    gap = self.compute_gap(pkt_id, q_val, last_hop)
                    if not (gap>= 0 and gap <= BaseDqnRoutingAgent.EXPERIENCE_RELEVANCY_PARAMS["gap_threshold"]):
                        send_signalling = 0
            else:
                send_signalling = 0 # send the signalling packet if gap_threshold is not activated or new packet
            
            ### store the transition in the temporary replay buffer
            if pkt_id not in BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index]:
                BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][pkt_id] = []
            BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][pkt_id].append({
                "obs": obs,
                "action": action,
                "previous_hop": self.index,
                "reward": None,
                "next_obs": None,
                "flag": None,
                "q_val": q_val,
                "acked": False,
                "action_time": BaseDqnRoutingAgent.logger.logging["episode_time"]
            })
        else:
            send_signalling = 0
        ## Increment the simulation and episode counters
        BaseDqnRoutingAgent.logger.logging["total_nb_iterations"] += 1
        BaseDqnRoutingAgent.logger.logging["nodes_nb_iterations"][self.index] += 1
        return action + send_signalling

    def compute_gap(self, pkt_id: (str), q_val: (list), last_hop: (int)) -> float:
        """ Compute the gap between the node and the neighbors estimations following the formula in the paper

        Args:
            pkt_id (str): packet id
            q_val (list): q values for actual node (the neighbor)
            last_hop (int): last hop

        Returns:
            float: gap value
        """
        info = BaseDqnRoutingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][-1]
        neighbor_estimation = float(info["q_val"][info["action"]])
        if info["flag"]:
            node_estimation = float(info["reward"])
        else:
            node_estimation = float(
                info["reward"] + BaseDqnRoutingAgent.RL_PARAMS["gamma"] * np.min(q_val).item()
            )
        gap = abs(neighbor_estimation - node_estimation) / node_estimation
        # write the estimations and the gap to the tensorboard
        # TODO : use tensorboard logger object to log the gap
        # with self.tb_writer_dict["estimation"].as_default():
        #     tf.summary.scalar(
        #         "neighbor_estimation",
        #         neighbor_estimation,
        #         step=int((BaseDqnRoutingAgent.shared + BaseDqnRoutingAgent.shared["curr_time"]) * 1e6),
        #     )
        #     tf.summary.scalar(
        #         "node_estimation",
        #         node_estimation,
        #         step=int((Agent.base_curr_time + Agent.curr_time) * 1e6),
        #     )
        #     tf.summary.scalar(
        #         "gap", gap, step=int((Agent.base_curr_time + Agent.curr_time) * 1e6)
        #     )
        BaseDqnRoutingAgent.logger.logging["gap"] = gap
        if node_estimation > 0: ### consider the gap only if the estimation of the node is positive
            return gap
        return -1
    
    def run(self):
        """Main method that will be threaded"""
        while True:
            obs = self.env.reset()
            if len(obs) > 1:
                obs = [obs[0]] +  list(np.array(obs[1:])/1000000)
            self.transition_number = 0
            pkt_id = None
            last_hop = -1
            done_flag = False
            info = ""
            while True:
                if not self.env.connected:
                    break
                ### compute the action for the given observation
                action = self.get_action(obs, pkt_id, last_hop)
                ### send the action to the simulator
                obs, _, done_flag, info = self.env.step(action)
                if len(obs) > 1:
                    if BaseDqnRoutingAgent.TOPOLOGY_PARAMS["monitoring_type"] == "passive":
                        obs = [obs[0]] + np.array(self.moving_avg_delays).mean(axis=1).tolist()
                    else:
                        obs = [obs[0]] +  list(np.array(obs[1:])/1000000)
                if info == "" :
                    print("info is empty")
                # print("****info", info)
                ### treat the given info from the env and do necessary actions
                pkt_id, last_hop = self.operate(info, obs, done_flag)

                ### train the agents if necessary
                if BaseDqnRoutingAgent.RL_PARAMS["train"]:
                    time_since_last_train = BaseDqnRoutingAgent.logger.logging["episode_time"]- BaseDqnRoutingAgent.shared["last_training_time"]
                    ### check if it is time to train the agent
                    if (time_since_last_train >= BaseDqnRoutingAgent.RL_PARAMS["training_step"]):
                        nb_training_steps = int(time_since_last_train/ BaseDqnRoutingAgent.RL_PARAMS["training_step"])
                        # print("start training", nb_training_steps, BaseDqnRoutingAgent.logger.logging["curr_time"], BaseDqnRoutingAgent.shared["last_training_time"])
                        self.train_agents(nb_training_steps)
                        BaseDqnRoutingAgent.shared["last_training_time"] = BaseDqnRoutingAgent.logger.logging["episode_time"]
                    ### check if it is time to sync the target neural network when model sharing is activated
                    if BaseDqnRoutingAgent.RL_PARAMS["agent_type"] in ("dqn_model_sharing", "dqn_logit_sharing", "madrl_centralized"):
                        self.sync_upcoming_target_nn()
                    ### log the data if necessary
                    ### check if it is time to log and training is activated
                    if BaseDqnRoutingAgent.SIMULATION_PARAMS["logging_timestep"] > 0 and BaseDqnRoutingAgent.RL_PARAMS["train"]:
                        if BaseDqnRoutingAgent.logger.logging["curr_time"] - BaseDqnRoutingAgent.logger.logging["last_log_time"] >= BaseDqnRoutingAgent.SIMULATION_PARAMS["logging_timestep"]:
                            self.log()
                            if BaseDqnRoutingAgent.logger.logging["curr_time"] <= BaseDqnRoutingAgent.SIMULATION_PARAMS["episode_duration"] - BaseDqnRoutingAgent.SIMULATION_PARAMS["logging_timestep"]:
                                self.logger.reset_moving_stats()
                            BaseDqnRoutingAgent.logger.logging["last_log_time"] = BaseDqnRoutingAgent.logger.logging["curr_time"]
                    ### update the lambda coefficients for the guided reward
                    if BaseDqnRoutingAgent.RL_PARAMS["loss_penalty_type"] == "guided":
                        if BaseDqnRoutingAgent.GUIDED_REWARD_PARAMS["lambda_train_step"] > 0:
                            time_since_last_lambda_train = BaseDqnRoutingAgent.logger.logging["episode_time"] - BaseDqnRoutingAgent.shared["last_lambda_training_time"]
                            if time_since_last_lambda_train >= BaseDqnRoutingAgent.GUIDED_REWARD_PARAMS["lambda_train_step"]:
                                for n in range(BaseDqnRoutingAgent.TOPOLOGY_PARAMS["nb_nodes"]):
                                    self.update_lambda_coefs(n)
                                BaseDqnRoutingAgent.shared["last_lambda_training_time"] = BaseDqnRoutingAgent.logger.logging["episode_time"]
                        
                            
                ## check if episode is done
                if obs[0] == -1 and done_flag:  # check if the episode is done
                    self.log()
                    BaseDqnRoutingAgent.logger.logging["last_log_time"] = BaseDqnRoutingAgent.logger.logging["curr_time"]
                    break
                ## check if the episode is done by max number of arrived pkts
                if (
                    BaseDqnRoutingAgent.SIMULATION_PARAMS["max_nb_arrived_pkts"] > 0
                    and BaseDqnRoutingAgent.SIMULATION_PARAMS["max_nb_arrived_pkts"]
                    <= BaseDqnRoutingAgent.logger.logging["total_arrived_pkts"]
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
    
    def operate(self, info: (str), obs: (list), flag: (bool)):
        """Treat the info from the env and fill the static vars
        Args:
            info (str): info received from the ns3 simulator
            obs (list): observation
            flag (bool): if True, the packet arrived at destination
        Returns:
            bool: True if it is a control packet, False otherwise
        """
        tokens = info.split(",")
        last_hop = -1
        ### retieve simulation time
        BaseDqnRoutingAgent.logger.logging["episode_time"] = float(tokens[2].split("=")[-1])
        BaseDqnRoutingAgent.logger.logging["curr_time"] = BaseDqnRoutingAgent.logger.logging[
            "episode_time"
        ] + (
            BaseDqnRoutingAgent.logger.logging["episode_index"]
            * BaseDqnRoutingAgent.SIMULATION_PARAMS["episode_duration"]
        )
        ### retrieve packet info
        delay_time = float(tokens[0].split("=")[-1])
        pkt_size = float(tokens[1].split("=")[-1])
        pkt_id = int(tokens[3].split("=")[-1])
        pkt_type = int(tokens[4].split("=")[-1])
        if pkt_type != 0:  # control packet
            if pkt_type == 1:  # target update packet
                # print("target update packet", tokens)
                nn_index = int(
                    tokens[20].split("=")[-1]
                )  # version of the neural network
                seg_index = int(tokens[21].split("=")[-1])  # segment index
                node_id_signaled = int(
                    tokens[22].split("=")[-1]
                )  # node id of the source node
                self.treat_target_update_pkt(nn_index, seg_index, node_id_signaled)
                BaseDqnRoutingAgent.logger.logging["target_update_overhead_size"] += pkt_size
                BaseDqnRoutingAgent.logger.logging["nb_target_update_pkts"] += 1
                BaseDqnRoutingAgent.logger.mv_overhead = pd.concat(
                    [
                        BaseDqnRoutingAgent.logger.mv_overhead,
                        pd.Series(
                            [pkt_size],
                            index=[BaseDqnRoutingAgent.logger.logging["curr_time"]],
                        ),
                    ]
                )
            if pkt_type == 2:  # replay memory update packet
                id_signaled = int(tokens[20].split("=")[-1]) # retieve the pkt id signaled
                arrived_from = int(tokens[21].split("=")[-1]) # retieve the node id of the source node
                is_first = int(tokens[22].split("=")[-1]) # retieve if it is the first segment
                self.treat_memory_update_pkt(id_signaled, arrived_from, is_first)
                BaseDqnRoutingAgent.logger.logging["replay_memory_update_overhead_size"] += pkt_size
                BaseDqnRoutingAgent.logger.logging["nb_memory_update_pkts"] += 1
                BaseDqnRoutingAgent.logger.mv_overhead = pd.concat(
                    [
                        BaseDqnRoutingAgent.logger.mv_overhead,
                        pd.Series(
                            [pkt_size],
                            index=[BaseDqnRoutingAgent.logger.logging["curr_time"]],
                        ),
                    ]
                )
        else:  # data packet
            ### retieve the lost packets
            lost_pkts_ids = (
                tokens[20].split("=")[-1].split(";")[:-1]
            )  # list of lost packets ids
            
            self.treat_lost_pkts(lost_pkts_ids)
            
            ### treat the arrived packet
            if obs[0] >= 0:
                last_hop = self.treat_arrived_packet(pkt_id, pkt_size, delay_time, obs, flag)
                
            ## update stats in static variables
            self.fill_stats(tokens)
            
        return pkt_id, last_hop

    def sync_upcoming_target_nn(self):
        """Sync the target neural network"""
        raise NotImplementedError("sync_upcoming_target_nn method is not implemented")
    
    def treat_lost_pkts(self, lost_pkts_ids: (list)):
        """Treat the lost packets
        Args:
            lost_pkts_ids (list): lost packet ids
        """        
        for pkt in lost_pkts_ids:
            lost_packet_info = BaseDqnRoutingAgent.shared["packets_in_network"][int(pkt)]
            BaseDqnRoutingAgent.logger.lost_packets_per_flow[lost_packet_info["src"]][lost_packet_info["dst"]] += 1
            ### Increment the loss counter
            BaseDqnRoutingAgent.logger.logging["notified_lost_pkts"] += 1
            BaseDqnRoutingAgent.logger.mv_avg_loss = pd.concat(
                [
                    BaseDqnRoutingAgent.logger.mv_avg_loss,
                    pd.Series(
                        1,
                        index=[BaseDqnRoutingAgent.logger.logging["curr_time"]],
                    ),
                ]
            )
        if BaseDqnRoutingAgent.TOPOLOGY_PARAMS["auto_detect_loss"]:
            lost_pkts_ids = []
            # consider the packets as lost if action_time is greater than the max delay
            for pkt_id in list(BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index].keys()):
                if BaseDqnRoutingAgent.shared["ports_timeout"][self.index] > 0:
                    if BaseDqnRoutingAgent.logger.logging["episode_time"] - BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][pkt_id][0]["action_time"] > BaseDqnRoutingAgent.shared["ports_timeout"][self.index]:
                        if pkt_id in BaseDqnRoutingAgent.shared["lost_detected"]:
                            BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][pkt_id].pop(0)
                            if len(BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][pkt_id]) == 0:
                                BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index].pop(pkt_id)
                        else:
                            if pkt_id in BaseDqnRoutingAgent.shared["packets_in_network"]:
                                if BaseDqnRoutingAgent.shared["packets_in_network"][int(pkt_id)]["node"] == self.index and len(BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][pkt_id]) == 1:
                                    lost_pkts_ids.append(pkt_id)
            BaseDqnRoutingAgent.shared["lost_detected"].extend(lost_pkts_ids)
            BaseDqnRoutingAgent.logger.logging["lost_pkts_detected"] = len(BaseDqnRoutingAgent.shared["lost_detected"])
            # print("auto detect loss", len(BaseDqnRoutingAgent.shared["lost_detected"]))

        for lost_packet_id in lost_pkts_ids:
            if (
                int(lost_packet_id) not in BaseDqnRoutingAgent.shared["packets_in_network"]
            ):  # check for a bug in the ns3 simulator
                # raise ValueError("Unknown lost packet id", lost_pkts_ids, lost_packet_id, int(lost_packet_id) in BaseDqnRoutingAgent.shared["packets_in_network"], BaseDqnRoutingAgent.logger.logging["episode_time"], BaseDqnRoutingAgent.shared["packets_in_network"].keys())
                continue
            BaseDqnRoutingAgent.shared["packets_in_network"].pop(
                int(lost_packet_id)
            )
            # lost_packet_info["lost_time"] = BaseDqnRoutingAgent.logger.logging["episode_time"]
            
            ### treat the lost packet when training
            if BaseDqnRoutingAgent.RL_PARAMS["train"]:
                ### store the transition in the replay buffer and remove it from the temporary replay buffer
                if int(lost_packet_id) in BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index]:
                    info = BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][int(lost_packet_id)].pop(0)
                    ### compute the reward
                    if BaseDqnRoutingAgent.RL_PARAMS["loss_penalty_type"] == "guided":
                        reward = -1 # to get replaced when training by the lambda coef
                        BaseDqnRoutingAgent.shared["nb_loss_pkts"][self.index][info["action"]] += 1
                    elif BaseDqnRoutingAgent.RL_PARAMS["loss_penalty_type"] == "fixed":
                        reward = BaseDqnRoutingAgent.RL_PARAMS["fixed_loss_penalty"]
                    if len(BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][int(lost_packet_id)]) == 0:
                        BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index].pop(int(lost_packet_id))
                    if BaseDqnRoutingAgent.RL_PARAMS["loss_penalty_type"] in ("guided", "fixed"):
                        BaseDqnRoutingAgent.shared["replay_buffers"][self.index].add(
                            info["obs"],
                            info["action"],
                            reward,
                            info["obs"][:1] + ([1] * len(list(BaseDqnRoutingAgent.TOPOLOGY_PARAMS["G"].neighbors(self.neighbors[info["action"]])))),
                            True,
                        )
                        ### add replay buffer length to the logs
                        BaseDqnRoutingAgent.logger.logging["replay_buffer_lengths"][self.index] = len(BaseDqnRoutingAgent.shared["replay_buffers"][self.index])
    
    def treat_memory_update_pkt(self, id_signaled: (int), arrived_from: (int), is_first: (int)):
        """Treat the replay memory update packet. This method is used to update the replay memory of the node by adding the transitions from the temporary replay buffer.
        Args:
            id_signaled (int): id of the packet signaled
            arrived_from (int): node id of the source node
            is_first (int): if 1, it is the first segment of the packet
        Returns:
            dict: info of the transition
        """
        if BaseDqnRoutingAgent.RL_PARAMS["train"]:
            ### extract the temporary replay buffer
            if id_signaled in BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index]:
                ## pop the first element
                info = BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][id_signaled].pop(0)
                if len(BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index][id_signaled]) == 0:
                    BaseDqnRoutingAgent.shared["temp_replay_buffers"][self.index].pop(id_signaled)
                if BaseDqnRoutingAgent.TOPOLOGY_PARAMS["auto_detect_loss"]:
                    # update the max delay
                    BaseDqnRoutingAgent.shared["ports_timeout"][self.index] = max(info["reward"]*2, BaseDqnRoutingAgent.shared["ports_timeout"][self.index])
                if BaseDqnRoutingAgent.TOPOLOGY_PARAMS["monitoring_type"] == "passive":
                    self.moving_avg_delays[info["action"]].append(info["reward"])
                    if len(self.moving_avg_delays[info["action"]]) > BaseDqnRoutingAgent.TOPOLOGY_PARAMS["monitoring_window"]:
                        self.moving_avg_delays[info["action"]].pop(0)
                if info["next_obs"] is None:
                    raise ValueError("next_obs is None")
                ### add the transition to the replay memory
                BaseDqnRoutingAgent.shared["replay_buffers"][self.index].add(
                    info["obs"],
                    info["action"],
                    info["reward"],
                    info["next_obs"],
                    info["flag"],
                )
                BaseDqnRoutingAgent.logger.logging["replay_buffer_lengths"][self.index] = len(BaseDqnRoutingAgent.shared["replay_buffers"][self.index])
                return info
        return None

    def treat_target_update_pkt(
        self, nn_index: (int), seg_index: (int), id_signaled: (int)
    ):
        """Treat the target update packet
        Args:
            nn_index (int): index of the neural network
            seg_index (int): index of the segment
            id_signaled (int): node id of the node signaled
        """
        raise NotImplementedError("treat_target_update_pkt method is not implemented")


    def train_agents(self, nb_training_steps: (int)):
        """Train the agents if it is time for them to train
        The agents are trained in parralel using threads
        Args:
            nb_training_steps (int): number of training steps
        """

        ### train the agents in parallel
        threads = []
        for i in range(BaseDqnRoutingAgent.TOPOLOGY_PARAMS["nb_nodes"]):
            threads.append(
                threading.Thread(
                    target=self.train, args=(nb_training_steps, i), daemon=True
                )
            )
            threads[-1].start()
            
    def log(self):
        """ Log data into tensorboard"""
        ### log the data
        super().log()
        if BaseDqnRoutingAgent.RL_PARAMS["train"]:
            BaseDqnRoutingAgent.logger.stats_writer_train()

        
    def train(self, nb_training_steps: (int), index: (int)):
        """Train the agent[agent_index]
        Args:
            nb_training_steps (int): number of training steps
            index (int): agent index
        """
        ### train the agent
        raise NotImplementedError("train method is not implemented")

    def close(self):
        """Close the node"""
        # super().close()
        ### intermediate episode end
        if BaseDqnRoutingAgent.logger.logging["episode_index"] != BaseDqnRoutingAgent.SIMULATION_PARAMS["nb_episodes"] - 1:
            ### update the lambda coefs when guided reward is used and lambda_train_step is not set
            if BaseDqnRoutingAgent.RL_PARAMS["train"] and BaseDqnRoutingAgent.RL_PARAMS["loss_penalty_type"] == "guided" and BaseDqnRoutingAgent.GUIDED_REWARD_PARAMS["lambda_train_step"] <= 0:
                self.update_lambda_coefs(self.index)
        else:
            ### save the models at the end of the simulation
            if BaseDqnRoutingAgent.RL_PARAMS["save_models"] and BaseDqnRoutingAgent.RL_PARAMS["train"]:
                self.save_model()
    
    def update_lambda_coefs(self, n: (int) = None):
        """Update the lambda coefs when guided reward is used
        Args:
            n (int): node index        
        """
        for i in range(len(list(BaseDqnRoutingAgent.TOPOLOGY_PARAMS["G"].neighbors(n)))):
            # TODO: think about reducing the lambda coefs
            BaseDqnRoutingAgent.shared["lambda_coefs"][n][i] += np.max([0, BaseDqnRoutingAgent.GUIDED_REWARD_PARAMS["lambda_lr"] * (BaseDqnRoutingAgent.shared["nb_loss_pkts"][n][i]- BaseDqnRoutingAgent.GUIDED_REWARD_PARAMS["alpha"])])
            BaseDqnRoutingAgent.shared["lambda_coefs"][n][i] = np.max([0, BaseDqnRoutingAgent.shared["lambda_coefs"][n][i]])
            BaseDqnRoutingAgent.shared["nb_loss_pkts"][n][i] = 0
            # log the lambda coefs
            BaseDqnRoutingAgent.logger.log_lambda_coefs(n, i, BaseDqnRoutingAgent.shared["lambda_coefs"][n][i])

    def save_model(self):
        """Save the model"""
        ## save models
        if BaseDqnRoutingAgent.RL_PARAMS["save_models"]:
            # path = BaseDqnRoutingAgent.RL_PARAMS["saved_models_path"].rstrip("/") + "/" + BaseDqnRoutingAgent.logger.session_name + "/episode_" + str(BaseDqnRoutingAgent.logger.logging["episode_index"]) + "/node_" + str(self.index)
            path = BaseDqnRoutingAgent.RL_PARAMS["saved_models_path"].rstrip("/") + "/" + BaseDqnRoutingAgent.logger.session_name + "/final"+ "/node_" + str(self.index)
            if not os.path.exists(path):
                os.makedirs(path)
            BaseDqnRoutingAgent.shared["DRL_agents"][self.index].save(path)
    

#     def save_model(
#     actor, node_index, path, t, nb_episodes, root="saved_models/", snapshot=False
# ):
#     """
#     Save the DQN model for a node into a folder.

#     Parameters
#     ----------
#     actors : DQN model
#         DQN model (for a network node).
#     node_index : int
#         index of the node.
#     path : str
#         name of the folder where to store the model.
#     t : int
#         number of passed train iterations
#     nb_episodes : int
#         number of passed episodes
#     root : str, optional
#         root folder where to store the model. The default is "saved_models/".
#     snapshot : bool, optional
#         if True, the model is saved in a folder named "episode_{nb_episodes}_step_{t}". The default is False.

#     Returns
#     -------
#     None.

#     """
#     if path.rstrip("/") not in os.listdir(root):
#         path = path + "/"
#         os.mkdir(root + path)
#     root = root.rstrip("/") + "/"
#     path = path.rstrip("/") + "/"
#     if snapshot:
#         folder_name = root + path + f"episode_{nb_episodes}_step_{t}"
#     else:
#         folder_name = root + path + "final"
#     actor.q_network.save(f"{folder_name}/node{node_index}")


# def save_all_models(
#     actors, overlay_nodes, path, t, nb_episodes, root="saved_models/", snapshot=False
# ):
#     """
#     Save all DQN models for each node into a folder.

#     Parameters
#     ----------
#     actors : list
#         list of DQN models (one for each network node).
#     path : str
#         name of the folder where to store the models.
#     t : int
#         number of passed train iterations
#     nb_episodes : int
#         number of passed episodes

#     Returns
#     -------
#     None.

#     """
#     for i in overlay_nodes:
#         save_model(actors[i], i, path, t, nb_episodes, root, snapshot)

