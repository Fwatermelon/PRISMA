__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2023 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"



import tensorflow as tf
import numpy as np
from source.controller.agents.base_dqn_routing import BaseDqnRoutingAgent

class ValueSharingAgent(BaseDqnRoutingAgent):
    """Base class for DQN routing
    The changes from the base class are:
    - Compute the target value when the next hop receives the packet and store it in the temporary replay buffer as the reward
    - Train the agent with the target value computed before
    
    Note: the target value is computed with the active neural network, so the neural network is used twice in the same step.
    Next step: use the target neural network to compute the target value. This will require to add a target neural network to the agent and regular updates of the target neural network.
    """

    # TODO: add double q learning with target network
    
    @classmethod
    def init_static_vars(cls, params_dict, logger):
        BaseDqnRoutingAgent.init_static_vars(params_dict, logger)
        if ValueSharingAgent.RL_PARAMS["train"]:
            params_dict["signaling_type"] = "target"

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
        if last_hop>=0 and ValueSharingAgent.RL_PARAMS["train"]:
            if pkt_id in ValueSharingAgent.shared["temp_replay_buffers"][last_hop]:
                for idx, key in enumerate(ValueSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id]):
                    if key["next_obs"] is None:
                        break
                ### store the transition in the temporary replay buffer
                ValueSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["next_obs"] = [ValueSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["obs"][0]] + list(obs[1:])
                ### get the neighbors of the neighbor and mask the action
                if ValueSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["previous_hop"] != last_hop:
                    raise ValueError("The previous hop is not the last hop")
                neighbors = self.neighbors[:]
                neighbors.remove(last_hop)
                ### replace the reward with the target q value
                ValueSharingAgent.shared["nn_locks"][self.index].acquire()
                ValueSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["reward"] = ValueSharingAgent.shared["DRL_agents"][self.index].get_target_value(delay_time, np.array([obs]), flag, neighbors).numpy().item()
                ValueSharingAgent.shared["nn_locks"][self.index].release()
                ValueSharingAgent.shared["temp_replay_buffers"][last_hop][pkt_id][idx]["flag"] = flag
        return last_hop

        
    def train(self, nb_training_steps: (int), index: (int)):
        """Train the agent[agent_index]
        Args:
            nb_training_steps (int): number of training steps
            index (int): agent index
        """
        ### train the agent
        ValueSharingAgent.shared["nn_locks"][index].acquire()
        td_errors = []
        if len(ValueSharingAgent.shared["replay_buffers"][index]) > ValueSharingAgent.RL_PARAMS["batch_size"]:
            for _ in range(nb_training_steps):
                ### get the batch from the temporary replay buffer
                obses_t, actions_t, targets_t, _, _, weights = ValueSharingAgent.shared["replay_buffers"][index].sample(ValueSharingAgent.RL_PARAMS["batch_size"])
                if ValueSharingAgent.loss_penalty_type == "guided":
                    # replace -1 by lambda coef for each output interface
                    for i in range(len(targets_t)):
                        targets_t[i] = [ValueSharingAgent.shared["lamda_coefs"][index][actions_t[i]] if x == -1 else x for x in targets_t[i]]
                ### train the agent
                td_error = ValueSharingAgent.shared["DRL_agents"][index].train(tf.constant(obses_t, dtype=tf.float32), tf.constant(actions_t, dtype=tf.int32), tf.constant(targets_t, dtype=tf.float32), tf.constant(weights, dtype=tf.float32))
                td_errors.append(tf.reduce_mean(td_error**2).numpy().item())
            ### save the td error
            ValueSharingAgent.logger.logging["td_errors"][index] = np.mean(td_errors).item()
        ValueSharingAgent.shared["nn_locks"][index].release()
