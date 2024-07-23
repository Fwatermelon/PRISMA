
__author__ = (
    "Redha A. Alliche"
)
__copyright__ = "Copyright (c) 2024 Redha A. Alliche"
__license__ = "GPL"
__email__ = "alliche@i3s.unice.fr"


import argparse
from source.controller.agents.base_dqn_routing import BaseDqnRoutingAgent
import numpy as np
import tensorflow as tf

class ModelSharingAgent(BaseDqnRoutingAgent):
    """Base class for DQN Model sharing"""
    
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
        group = parser.add_argument_group("Model Sharing arguments")

        group.add_argument(
            "--nn_size",
            type=int,
            help="Size of the neural network in bytes (used when signaling type is NN)",
            default=35328,
        )
        group.add_argument(
            "--target_update_period",
            type=float,
            help="Target update period U for Model sharing (used when training)",
            default=1.0,
        )
        return parser

    @classmethod
    def init_static_vars(cls, params_dict, logger):
        BaseDqnRoutingAgent.init_static_vars(params_dict, logger)
        if ModelSharingAgent.RL_PARAMS["train"]:
            params_dict["signaling_type"] = "NN"
            ModelSharingAgent.shared["nn_max_seg_index"] = params_dict["nn_size"] // params_dict["packet_size"]
            ModelSharingAgent.shared["sync_counter"] = 0
            ModelSharingAgent.RL_PARAMS["target_update_period"] = params_dict["target_update_period"]


    def reset_node(self):
        super().reset_node()
        self.sync_segment_counters = [1 for _ in range(len(self.neighbors))]
        ModelSharingAgent.shared["sync_counter"] = 0
        
    def treat_target_update_pkt(
        self, nn_index: (int), seg_index: (int), id_signaled: (int)
    ):
        """Treat the target update packet
        Args:
            nn_index (int): index of the neural network
            seg_index (int): index of the segment
            id_signaled (int): node id of the node signaled
        """
        if ModelSharingAgent.RL_PARAMS["train"]:
            sync_counter = ModelSharingAgent.shared["sync_counter"]-1
            neighbor_idx = self.neighbors.index(id_signaled)
            if seg_index > ModelSharingAgent.shared["nn_max_seg_index"]:
                raise ValueError("seg_index > {}".format(ModelSharingAgent.shared["nn_max_seg_index"]))
            if nn_index > sync_counter and int(ModelSharingAgent.logger.logging["episode_time"]) > 0:
                raise ValueError("nn_index > {}. Should sync upcoming".format(sync_counter))
            elif nn_index < sync_counter-1:
                raise ValueError("nn_index < {}-1. Too late communication".format(sync_counter-2))
            # check if the signaling have not been reset
            else:
                ## check if the signaling is complete
                if seg_index == self.sync_segment_counters[neighbor_idx]:
                    if self.sync_segment_counters[neighbor_idx] == ModelSharingAgent.shared["nn_max_seg_index"]-1:
                        if nn_index == sync_counter-1:
                            temp = True
                        else:
                            temp = False
                        self.sync_current(neighbor_idx, with_temp=temp)
                        self.sync_segment_counters[neighbor_idx] = 0
                    else:
                        self.sync_segment_counters[neighbor_idx] += 1
                else:
                    self.sync_segment_counters[neighbor_idx] = 0

    def sync_current(self, neighbor_idx: (int), with_temp: (bool) = False):
        """
        Sync this node neighbor target neural network with the upcoming target nn

        Args:
            neighbor_idx (int): neighbor index for this node
        """
        print("sync current", self.index, neighbor_idx, ModelSharingAgent.logger.logging["episode_time"])
        ModelSharingAgent.shared["nn_locks"][self.index].acquire()
        ModelSharingAgent.shared["DRL_agents"][self.index].sync_neighbor_target_q_network(
            neighbor_idx, with_temp=with_temp
        )
        ModelSharingAgent.shared["nn_locks"][self.index].release()
    
    def sync_upcoming_target_nn(self):
        """
        Check if it is time to sync the target neural network of the neighbor in the upcoming nn
        """
        if (
            ModelSharingAgent.shared["sync_counter"] *
            ModelSharingAgent.RL_PARAMS["target_update_period"] <= ModelSharingAgent.logger.logging["episode_time"]
        ):
            for i in range(ModelSharingAgent.TOPOLOGY_PARAMS["nb_nodes"]):
                for neighbor_idx, neighbor in enumerate(list(ModelSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(i))):
                    # ModelSharingAgent.shared["DRL_agents"][i].sync_neighbor_upcoming_target_q_network(ModelSharingAgent.shared["DRL_agents"][neighbor], neighbor_idx)
                    # TODO offband
                    # ModelSharingAgent.shared["DRL_agents"][i].sync_neighbor_target_q_network(ModelSharingAgent.shared["DRL_agents"][neighbor], neighbor_idx)
                    q_vars = ModelSharingAgent.shared["DRL_agents"][neighbor].q_network.trainable_variables
                    target_q_vars = ModelSharingAgent.shared["DRL_agents"][i].neighbors_target_q_networks[neighbor_idx].trainable_variables
                    for var, var_target in zip(q_vars, target_q_vars):
                        var_target.assign(var)
            ModelSharingAgent.shared["sync_counter"] += 1
        
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
            flag (bool): if True, the packet arrived at destination
        Returns:
            last_hop (int): last hop
        """
        ### treat new arrived packet
        last_hop = super().treat_arrived_packet(pkt_id, pkt_size, delay_time, obs, flag)
        
        ### treat the hop packet when training to add it to complete the information in the temporary replay buffer
        if last_hop>=0 and ModelSharingAgent.RL_PARAMS["train"]:
            if pkt_id in ModelSharingAgent.shared["temp_replay_buffers"][last_hop]:
                for idx, key in enumerate(ModelSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id]):
                    if key["next_obs"] is None:
                        break
                ### store the transition in the temporary replay buffer
                ModelSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["next_obs"] = obs
                ### get the neighbors of the neighbor and mask the action
                if ModelSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["previous_hop"] != last_hop:
                    raise ValueError("The previous hop is not the last hop")
                ModelSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["reward"] = delay_time
                ModelSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["flag"] = flag
        return last_hop

    def train(self, nb_training_steps: (int), index: (int)):
        """Train the agent[agent_index]
        Args:
            nb_training_steps (int): number of training steps
            index (int): agent index
        """
        ### train the agent
        td_errors = []
        ModelSharingAgent.shared["nn_locks"][index].acquire()
        if len(ModelSharingAgent.shared["replay_buffers"][index]) > ModelSharingAgent.RL_PARAMS["batch_size"]:
            neighbors = list(ModelSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(index))
            for _ in range(nb_training_steps):
                ### get the batch from the temporary replay buffer
                obses_t, actions_t, rewards_t, next_obses_t, flags_t, weights = ModelSharingAgent.shared["replay_buffers"][index].sample(ModelSharingAgent.RL_PARAMS["batch_size"])
                ### compute the target q values using the locally stored neighbors nns
                targets_t = []
                action_indices_all = []
                for indx, neighbor in enumerate(neighbors):
                    neighbors_of_neighbor = list(ModelSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(neighbor))
                    filtered_indices = np.where(np.array(neighbors_of_neighbor)!=index)[0] # filter the net interface from where the pkt comes
                    # filtered_indices = np.where(np.array(list(Agent.G.neighbors(neighbor)))!=1000)[0] # filter the net interface from where the pkt comes
                    action_indices = np.where(actions_t == indx)[0]
                    action_indices_all.append(action_indices)
                    if len(action_indices):
                        # penalty = 0
                        if ModelSharingAgent.loss_penalty_type == "guided":
                            # replace -1 by lambda coef
                            rewards_t[action_indices] = np.where(rewards_t[action_indices] == -1, ModelSharingAgent.shared["lamda_coefs"][index][indx], rewards_t[action_indices])
                        targets_t.append(ModelSharingAgent.shared["DRL_agents"][index].get_neighbor_target_value(indx, 
                                                                                            rewards_t[action_indices], 
                                                                                            tf.constant(np.array(np.vstack(next_obses_t[action_indices]),
                                                                                                                dtype=float)), 
                                                                                            flags_t[action_indices],
                                                                                            filtered_indices))
                action_indices_all = np.concatenate(action_indices_all)
                obses_t = tf.constant(obses_t[action_indices_all,])
                actions_t = tf.constant(actions_t[action_indices_all], shape=(ModelSharingAgent.RL_PARAMS["batch_size"]))
                targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(ModelSharingAgent.RL_PARAMS["batch_size"]))
                ### train the agent
                td_error = ModelSharingAgent.shared["DRL_agents"][index].train(tf.cast(obses_t, tf.float32), tf.cast(actions_t,tf.int32), tf.cast(targets_t, tf.float32), tf.constant(weights))
                td_errors.append(tf.reduce_mean(td_error**2).numpy().item())
            ### save the td error
            ModelSharingAgent.logger.logging["td_errors"][index] = np.mean(td_errors).item()
        ModelSharingAgent.shared["nn_locks"][index].release()