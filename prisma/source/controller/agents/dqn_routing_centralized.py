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

class CentralizedMADRLAgent(BaseDqnRoutingAgent):
    """Base class for Mulit-Agent DQN Routing Agent Model sharing in centralized training"""
    
    @classmethod
    def add_arguments(
        cls, parser: (argparse.ArgumentParser)
    ) -> argparse.ArgumentParser:
        """Argument parser for this class. This method is used to add new arguments to the global parser
        Target update periode is used as a Synchronization period in seconds, when the centralized controller should send the target neural network to the agents
        Args:
            parser (argparse.ArgumentParser): parser to add arguments to
        Returns:
            argparse.ArgumentParser: parser with added arguments
        """
        group = parser.add_argument_group("Model Sharing centralized settiong arguments")

        group.add_argument(
            "--controller_id",
            type=int,
            help="The id of the controller node (used when signaling type is centralized)",
            default=-1,
        )
        return parser

    @classmethod
    def init_static_vars(cls, params_dict, logger):
        BaseDqnRoutingAgent.init_static_vars(params_dict, logger)
        if CentralizedMADRLAgent.RL_PARAMS["train"]:
            params_dict["signaling_type"] = "centralized"
            CentralizedMADRLAgent.shared["nn_max_seg_index"] = params_dict["nn_size"] // params_dict["packet_size"]
            CentralizedMADRLAgent.shared["sync_counter"] = 0
            if params_dict["controller_id"] < 0:
                raise ValueError("The controller node id is not set")
            if params_dict["target_update_period"] <= 0:
                raise ValueError("The sync period should be positive")
            # check if the controller id is in the topology
            if params_dict["controller_id"] not in CentralizedMADRLAgent.TOPOLOGY_PARAMS["G"].nodes:
                raise ValueError("The controller node is not in the topology")
            if params_dict["controller_id"] == -1:
                raise ValueError("The controller node id is not set")
            CentralizedMADRLAgent.RL_PARAMS["controller_id"] = params_dict["controller_id"]
            CentralizedMADRLAgent.EXPERIENCE_RELEVANCY_PARAMS["gap_threshold"] = 0.0
            CentralizedMADRLAgent.RL_PARAMS["target_update_period"] = params_dict["target_update_period"]
            ## load target neural network and upcoming target neural network
            for i in range(CentralizedMADRLAgent.TOPOLOGY_PARAMS["nb_nodes"]):
                q_vars = cls.shared["DRL_agents"][i].q_network.trainable_variables
                target_q_vars = cls.shared["DRL_agents"][i].target_q_network.trainable_variables
                for var, var_target in zip(q_vars, target_q_vars):
                    var_target.assign(var)
                q_vars = cls.shared["DRL_agents"][i].q_network.trainable_variables
                target_q_vars = cls.shared["DRL_agents"][i].upcoming_target_q_network.trainable_variables
                for var, var_target in zip(q_vars, target_q_vars):
                    var_target.assign(var)




    def reset_node(self):
        super().reset_node()
        self.sync_segment_counters = [1 for _ in range(len(CentralizedMADRLAgent.TOPOLOGY_PARAMS["G"].nodes))]
        CentralizedMADRLAgent.shared["sync_counter"] = 0
        
        
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
            if id_signaled in BaseDqnRoutingAgent.shared["temp_replay_buffers"][arrived_from]:
                if BaseDqnRoutingAgent.shared["temp_replay_buffers"][arrived_from][id_signaled][0]["acked"]:
                    ## pop the first element
                    info = BaseDqnRoutingAgent.shared["temp_replay_buffers"][arrived_from][id_signaled].pop(0)
                    if len(BaseDqnRoutingAgent.shared["temp_replay_buffers"][arrived_from][id_signaled]) == 0:
                        BaseDqnRoutingAgent.shared["temp_replay_buffers"][arrived_from].pop(id_signaled)
                    if info["next_obs"] is None:
                        raise ValueError("next_obs is None")
                    ### add the transition to the replay memory
                    BaseDqnRoutingAgent.shared["replay_buffers"][arrived_from].add(
                        info["obs"],
                        info["action"],
                        info["reward"],
                        info["next_obs"],
                        info["flag"],
                    )
                    BaseDqnRoutingAgent.logger.logging["replay_buffer_lengths"][arrived_from] = len(BaseDqnRoutingAgent.shared["replay_buffers"][arrived_from])
                    return info
                else:
                    BaseDqnRoutingAgent.shared["temp_replay_buffers"][arrived_from][id_signaled][0]["acked"] = True                        
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
        if CentralizedMADRLAgent.RL_PARAMS["train"]:
            sync_counter = CentralizedMADRLAgent.shared["sync_counter"]-1
            if seg_index > CentralizedMADRLAgent.shared["nn_max_seg_index"]:
                raise ValueError("seg_index > {}".format(CentralizedMADRLAgent.shared["nn_max_seg_index"]))
            # if nn_index > sync_counter and int(CentralizedMADRLAgent.logger.logging["episode_time"]) > 0:
            #     raise ValueError("nn_index > {}. Should sync upcoming".format(sync_counter))
            elif nn_index < sync_counter-1:
                raise ValueError("nn_index < {}-1. Too late communication".format(sync_counter-2))
            # check if the signaling have not been reset
            else:
                ## check if the signaling is complete
                if seg_index == self.sync_segment_counters:
                    if self.sync_segment_counters == CentralizedMADRLAgent.shared["nn_max_seg_index"]-1:
                        self.sync_current()
                        self.logger.logging["sync_counter"] += 1
                        self.sync_segment_counters = 0
                    else:
                        self.sync_segment_counters += 1
                else:
                    self.sync_segment_counters = 0

    def sync_current(self):
        """
        Sync this node target neural network with the upcoming target nn

        Args:
            neighbor_idx (int): neighbor index for this node
        """
        CentralizedMADRLAgent.shared["nn_locks"][self.index].acquire()
        q_vars = CentralizedMADRLAgent.shared["DRL_agents"][self.index].upcoming_target_q_network.trainable_variables
        target_q_vars = CentralizedMADRLAgent.shared["DRL_agents"][self.index].q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)
        CentralizedMADRLAgent.shared["nn_locks"][self.index].release()
    
    def sync_upcoming_target_nn(self):
        """
        Check if it is time to sync the target neural network in the upcoming nn
        """
        if (
            CentralizedMADRLAgent.shared["sync_counter"] *
            CentralizedMADRLAgent.RL_PARAMS["target_update_period"] <= CentralizedMADRLAgent.logger.logging["episode_time"]
        ):
            for i in range(CentralizedMADRLAgent.TOPOLOGY_PARAMS["nb_nodes"]):
                # CentralizedMADRLAgent.shared["DRL_agents"][i].sync_neighbor_upcoming_target_q_network(CentralizedMADRLAgent.shared["DRL_agents"][neighbor], neighbor_idx)
                # TODO offband
                # CentralizedMADRLAgent.shared["DRL_agents"][i].sync_neighbor_target_q_network(CentralizedMADRLAgent.shared["DRL_agents"][neighbor], neighbor_idx)
                if i == CentralizedMADRLAgent.RL_PARAMS["controller_id"]:
                    q_vars = CentralizedMADRLAgent.shared["DRL_agents"][i].target_q_network.trainable_variables
                    target_q_vars = CentralizedMADRLAgent.shared["DRL_agents"][i].q_network.trainable_variables
                    for var, var_target in zip(q_vars, target_q_vars):
                        var_target.assign(var)
                    self.logger.logging["sync_counter"] += 1
                else:
                    q_vars = CentralizedMADRLAgent.shared["DRL_agents"][i].target_q_network.trainable_variables
                    target_q_vars = CentralizedMADRLAgent.shared["DRL_agents"][i].upcoming_target_q_network.trainable_variables
                    for var, var_target in zip(q_vars, target_q_vars):
                        var_target.assign(var)
                    
            CentralizedMADRLAgent.shared["sync_counter"] += 1

    
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
        if last_hop>=0 and CentralizedMADRLAgent.RL_PARAMS["train"]:
            if pkt_id in CentralizedMADRLAgent.shared["temp_replay_buffers"][last_hop]:
                for idx, key in enumerate(CentralizedMADRLAgent.shared["temp_replay_buffers"][last_hop][pkt_id]):
                    if key["next_obs"] is None:
                        break
                ### store the transition in the temporary replay buffer
                CentralizedMADRLAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["next_obs"] = obs
                ### get the neighbors of the neighbor and mask the action
                if CentralizedMADRLAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["previous_hop"] != last_hop:
                    raise ValueError("The previous hop is not the last hop")
                CentralizedMADRLAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["reward"] = delay_time
                CentralizedMADRLAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["flag"] = flag
        return last_hop

    def train(self, nb_training_steps: (int), index: (int)):
        """Train the agent[agent_index]
        Args:
            nb_training_steps (int): number of training steps
            index (int): agent index
        """
        ### train the agent
        td_errors = []
        CentralizedMADRLAgent.shared["nn_locks"][index].acquire()
        if len(CentralizedMADRLAgent.shared["replay_buffers"][index]) > CentralizedMADRLAgent.RL_PARAMS["batch_size"]:
            neighbors = list(CentralizedMADRLAgent.TOPOLOGY_PARAMS["G"].neighbors(index))
            for _ in range(nb_training_steps):
                ### get the batch from the temporary replay buffer
                obses_t, actions_t, rewards_t, next_obses_t, flags_t, weights = CentralizedMADRLAgent.shared["replay_buffers"][index].sample(CentralizedMADRLAgent.RL_PARAMS["batch_size"])
                ### compute the target q values using the locally stored neighbors nns
                targets_t = []
                action_indices_all = []
                for indx, neighbor in enumerate(neighbors):
                    neighbors_of_neighbor = list(CentralizedMADRLAgent.TOPOLOGY_PARAMS["G"].neighbors(neighbor))
                    filtered_indices = np.where(np.array(neighbors_of_neighbor)!=index)[0] # filter the net interface from where the pkt comes
                    # filtered_indices = np.where(np.array(list(Agent.G.neighbors(neighbor)))!=1000)[0] # filter the net interface from where the pkt comes
                    action_indices = np.where(actions_t == indx)[0]
                    action_indices_all.append(action_indices)
                    if len(action_indices):
                        # penalty = 0
                        if CentralizedMADRLAgent.RL_PARAMS["loss_penalty_type"] == "guided":
                            # replace -1 by lambda coef
                            rewards_t[action_indices] = np.where(rewards_t[action_indices] == -1, CentralizedMADRLAgent.shared["lambda_coefs"][index][indx], rewards_t[action_indices])
                        targets_t.append(CentralizedMADRLAgent.shared["DRL_agents"][neighbor].get_target_value(rewards_t[action_indices], tf.constant(np.array(np.vstack(next_obses_t[action_indices]), dtype=float)), flags_t[action_indices], filtered_indices, use_target=True))
                action_indices_all = np.concatenate(action_indices_all)
                obses_t = tf.constant(obses_t[action_indices_all,])
                actions_t = tf.constant(actions_t[action_indices_all], shape=(CentralizedMADRLAgent.RL_PARAMS["batch_size"]))
                targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(CentralizedMADRLAgent.RL_PARAMS["batch_size"]))
                ### train the agent
                td_error = CentralizedMADRLAgent.shared["DRL_agents"][index].train(tf.cast(obses_t, tf.float32), tf.cast(actions_t,tf.int32), tf.cast(targets_t, tf.float32), tf.constant(weights), use_target=True)
                td_errors.append(tf.reduce_mean(td_error**2).numpy().item())
            ### save the td error
            CentralizedMADRLAgent.logger.logging["td_errors"][index] = np.mean(td_errors).item()
        CentralizedMADRLAgent.shared["nn_locks"][index].release()