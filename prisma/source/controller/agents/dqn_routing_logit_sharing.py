__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2023 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"


import argparse
import numpy as np
import threading
from tensorflow import keras
import tensorflow as tf
from source.controller.agents.base_dqn_routing import BaseDqnRoutingAgent

class LogitSharingAgent(BaseDqnRoutingAgent):
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
        # TODO: group the argument of the logit sharing
        group = parser.add_argument_group("Logit sharing arguments")

        group.add_argument(
            "--d_t_send_all_destinations",
            type=int,
            help="If true, send the logits (output of the neighbor NN) for all the possible destinations",
            default=0,
        )
        return parser

    @classmethod
    def init_static_vars(cls, params_dict, logger):
        """Takes the parameters of the simulation from a dict and assign it values to the static vars"""
        BaseDqnRoutingAgent.init_static_vars(params_dict, logger)
        if LogitSharingAgent.RL_PARAMS["train"]:
            params_dict["signaling_type"] = "logit"
            LogitSharingAgent.RL_PARAMS["d_t_send_all_destinations"] = params_dict["d_t_send_all_destinations"]
            LogitSharingAgent.shared["sync_counter"] = 0
            LogitSharingAgent.RL_PARAMS["target_update_period"] = params_dict["target_update_period"]
            ### load the neighbors target neural networks
            for i in range(LogitSharingAgent.TOPOLOGY_PARAMS["nb_nodes"]):
                for ix, neighbor in enumerate(list(LogitSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(i))):
                    q_vars = cls.shared["DRL_agents"][neighbor].q_network.trainable_variables
                    target_q_vars = cls.shared["DRL_agents"][i].neighbors_d_t_network[ix].trainable_variables
                    for var, var_target in zip(q_vars, target_q_vars):
                        var_target.assign(var)

    def reset_node(self):
        """Reset the node when a new episode starts"""
        ### connect to the ns3 simulator
        super().reset_node()
        LogitSharingAgent.shared["sync_counter"] = 0
        

    def treat_memory_update_pkt(self, id_signaled: (int), arrived_from, is_first):
        """Treat the memory update packet
        Args:
            id_signaled (int): id of the node that signaled the memory update
        """
        info = super().treat_memory_update_pkt(id_signaled, arrived_from, is_first)
        if LogitSharingAgent.RL_PARAMS["train"]:
            ### add elements to supervised dataset
            if info is not None:
                LogitSharingAgent.shared["DRL_agents"][self.index].neighbors_d_t_database[info["action"]].add(info["next_obs"], info["next_q_val"], LogitSharingAgent.logger.logging["episode_time"])
        

    def sync_upcoming_target_nn(self):
        """
        Check if it is time to sync and train the digital twins in different threads
        """
        if (
            LogitSharingAgent.shared["sync_counter"] *
            LogitSharingAgent.RL_PARAMS["target_update_period"] <= LogitSharingAgent.logger.logging["episode_time"]
        ):
            for i in range(LogitSharingAgent.TOPOLOGY_PARAMS["nb_nodes"]):
                for neighbor_idx, neighbor in enumerate(list(LogitSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(i))):
                    thread = threading.Thread(target=self.train_d_t_, args=(i, neighbor_idx))
                    thread.start()
                self.logger.logging["sync_counter"] += 1
            LogitSharingAgent.shared["sync_counter"] += 1

    
    def train_d_t_(self, index, neighbor_idx):
        """
        Train the digital twin for a given neighbor
        """
        x, y = LogitSharingAgent.shared["DRL_agents"][index].neighbors_d_t_database[neighbor_idx].get_data()
        if len(y) == 0 or len(x) == 0:
            return
        size = min(len(x), len(y))
        # with Agent.train_lock:
        LogitSharingAgent.shared["nn_locks"][index].acquire()
        loss = LogitSharingAgent.shared["DRL_agents"][index].neighbors_d_t_network[neighbor_idx].fit(x[:size], y[:size], batch_size=LogitSharingAgent.RL_PARAMS["batch_size"], epochs=int(100*LogitSharingAgent.RL_PARAMS["target_update_period"]), verbose=0, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=1e-6)])
        print("supervised training; node = ", index, " neighbor = ", neighbor_idx, " loss = ", loss.history["loss"][-1] , " time = ", LogitSharingAgent.logger.logging["episode_time"], " len = ", len(y), len(x), " epochs = ", len(loss.history["loss"]))
        LogitSharingAgent.shared["nn_locks"][index].release()
    
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
        if last_hop>=0 and LogitSharingAgent.RL_PARAMS["train"]:
            if pkt_id in LogitSharingAgent.shared["temp_replay_buffers"][last_hop]:
                for idx, key in enumerate(LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id]):
                    if key["next_obs"] is None:
                        break
                ### store the transition in the temporary replay buffer
                if LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["obs"][0] != obs[0]:
                    raise ValueError("The observation is not the same")
                LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["next_obs"] = [LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["obs"][0]] + list(obs[1:])
                ### get the neighbors of the neighbor and mask the action
                if LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["previous_hop"] != last_hop:
                    raise ValueError("The previous hop is not the last hop")
                LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["reward"] = delay_time
                LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["flag"] = flag
                LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["next_q_val"] = LogitSharingAgent.shared["DRL_agents"][self.index].q_network(np.array([LogitSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["next_obs"]])).numpy().tolist()[0]
        return last_hop

    def train(self, nb_training_steps: (int), index: (int)):
        """Train the agent[agent_index]
        Args:
            nb_training_steps (int): number of training steps
            index (int): agent index
        """
        ### train the agent
        td_errors = []
        LogitSharingAgent.shared["nn_locks"][index].acquire()
        if len(LogitSharingAgent.shared["replay_buffers"][index]) > LogitSharingAgent.RL_PARAMS["batch_size"]:
            neighbors = list(LogitSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(index))
            for _ in range(nb_training_steps):
                ### get the batch from the temporary replay buffer
                obses_t, actions_t, rewards_t, next_obses_t, flags_t, weights = LogitSharingAgent.shared["replay_buffers"][index].sample(LogitSharingAgent.RL_PARAMS["batch_size"])
                ### compute the target q values using the locally stored neighbors nns
                targets_t = []
                action_indices_all = []
                for indx, neighbor in enumerate(neighbors):
                    neighbors_of_neighbor = list(LogitSharingAgent.TOPOLOGY_PARAMS["G"].neighbors(neighbor))
                    filtered_indices = np.where(np.array(neighbors_of_neighbor)!=index)[0] # filter the net interface from where the pkt comes
                    # filtered_indices = np.where(np.array(neighbors_of_neighbor)!=1000)[0] # filter the net interface from where the pkt comes
                    action_indices = np.where(actions_t == indx)[0]
                    action_indices_all.append(action_indices)
                    if len(action_indices):
                        if LogitSharingAgent.RL_PARAMS["loss_penalty_type"] == "guided":
                            # replace -1 by lambda coef
                            rewards_t[action_indices] = np.where(rewards_t[action_indices] == -1, LogitSharingAgent.shared["lambda_coefs"][index][indx], rewards_t[action_indices])
                        targets_t.append(LogitSharingAgent.shared["DRL_agents"][index].get_neighbor_d_t_value(indx, 
                                                                                            rewards_t[action_indices], 
                                                                                            tf.constant(np.array(np.vstack(next_obses_t[action_indices]),
                                                                                                                dtype=float)), 
                                                                                            flags_t[action_indices],
                                                                                            filtered_indices))
                action_indices_all = np.concatenate(action_indices_all)
                obses_t = tf.constant(obses_t[action_indices_all,])
                actions_t = tf.constant(actions_t[action_indices_all], shape=(LogitSharingAgent.RL_PARAMS["batch_size"]))
                targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(LogitSharingAgent.RL_PARAMS["batch_size"]))
                ### train the agent
                td_error = LogitSharingAgent.shared["DRL_agents"][index].train(tf.cast(obses_t, tf.float32), tf.cast(actions_t,tf.int32), tf.cast(targets_t, tf.float32), tf.constant(weights))
                td_errors.append(tf.reduce_mean(td_error**2).numpy().item())
            ### save the td error
            LogitSharingAgent.logger.logging["td_errors"][index] = np.mean(td_errors).item()
        LogitSharingAgent.shared["nn_locks"][index].release()
