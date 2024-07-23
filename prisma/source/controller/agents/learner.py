"""Deep Q model

The functions in this model:

======= step ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: tensor
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.

======= train =======

    Function that takes a transition (s,a,r,s',d) and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * (1-d) * min_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * min_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import os
import numpy as np
import tensorflow as tf
from source.controller.agents.replay_buffer import DigitalTwinDB

__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

# This work was partially based on openai/baselines available on https://github.com/openai/baselines/tree/tf2/baselines/deepq


# tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta, tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta)
    )

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def load_model(path, node_index=-1):
    """
    Loads the list of agents from a directory

    Parameters
    ----------
    path : str
        The folder containing the saved models.

    Returns
    -------
    a list of Q functions for each agent. Return None if an error occurs

    """
    from source.controller.agents.neural_nets_architectures import SplitLayer

    folders = os.listdir(path)
    q_functions = [1] * len(folders)
    for item in folders:
        index = int(str(item).lstrip("node").lstrip("_"))
        if node_index >= 0 and node_index != index:
            continue
        try:
            q_functions[index] = tf.keras.models.load_model(
                path + "/" + item,
                custom_objects={
                    "K": tf.keras.backend,
                    "layers": tf.keras.layers,
                    "SplitLayer": SplitLayer,
                    "tf": tf,
                },
                compile=False,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(e)
    return q_functions

#     if not(path.rstrip("/") in os.listdir(root)):
#         path = path + '/'
# =======
#     if not(path in os.listdir(root)):
# >>>>>>> 7ba840121a9f88c99c702aa70bc103e7c4769b00
#         os.mkdir(root + path)
#     root = root.rstrip('/') + '/'
#     path = path.rstrip('/') + '/'
#     if snapshot: 
#         folder_name = root + path + f"episode_{num_episodes}_step_{t}"
#     else:
#         folder_name = root + path + "final"
#     actor.q_network.save(f"{folder_name}/node{node_index}")

class DQN_AGENT(tf.Module):
    def __init__(
        self,
        q_func,
        observation_shape,
        num_actions,
        nb_nodes,
        lr,
        input_size_splits,
        neighbors_degrees,
        grad_norm_clipping=None,
        gamma=1.0,
        double_q=False,
        d_t_max_time=10,
        d_q_func=None,
    ):
        self.num_actions = num_actions
        self.q_func = q_func
        self.nb_nodes = nb_nodes
        self.gamma = gamma
        self.double_q = double_q
        self.grad_norm_clipping = grad_norm_clipping
        self.observation_shape = observation_shape
        self.input_size_splits = input_size_splits

        self.optimizer = tf.keras.optimizers.Adam(lr)

        with tf.name_scope("q_network"):
            self.q_network = q_func(
                observation_shape, num_actions, nb_nodes, input_size_splits
            )
        with tf.name_scope("target_q_network"):
            self.target_q_network = q_func(
                observation_shape, num_actions, nb_nodes, input_size_splits
            )
        self.eps = tf.Variable(0.0, name="eps")

        self.loss = tf.keras.losses.MeanSquaredError()
        ### define digital twin max time in seconds for the database
        self.d_t_max_time = d_t_max_time

        ### define the neighbors target q networks
        self.neighbors_target_q_networks = []
        self.neighbors_target_upcoming_q_networks = []
        self.neighbors_target_temp_upcoming_q_networks = []

        ### define the digital twin q networks
        self.neighbors_d_t_network = []

        ### define the digital twin database
        self.neighbors_d_t_database = []
        for neighbor in range(num_actions):
            with tf.name_scope(f"neighbor_target_q_network_{neighbor}"):
                self.neighbors_target_q_networks.append(
                    q_func(
                        (
                            neighbors_degrees[neighbor]
                            + observation_shape[0]
                            - num_actions,
                        ),
                        neighbors_degrees[neighbor],
                        nb_nodes,
                        [
                            1,
                            neighbors_degrees[neighbor],
                            observation_shape[0] - num_actions - 1,
                        ],
                    )
                )
            with tf.name_scope(f"neighbors_target_upcoming_q_network_{neighbor}"):
                self.neighbors_target_upcoming_q_networks.append(
                    q_func(
                        (
                            neighbors_degrees[neighbor]
                            + observation_shape[0]
                            - num_actions,
                        ),
                        neighbors_degrees[neighbor],
                        nb_nodes,
                        [
                            1,
                            neighbors_degrees[neighbor],
                            observation_shape[0] - num_actions - 1,
                        ],
                    )
                )
            with tf.name_scope(f"neighbors_target_temp_upcoming_q_network_{neighbor}"):
                self.neighbors_target_temp_upcoming_q_networks.append(
                    q_func(
                        (
                            neighbors_degrees[neighbor]
                            + observation_shape[0]
                            - num_actions,
                        ),
                        neighbors_degrees[neighbor],
                        nb_nodes,
                        [
                            1,
                            neighbors_degrees[neighbor],
                            observation_shape[0] - num_actions - 1,
                        ],
                    )
                )
            if d_q_func is not None:
                with tf.name_scope(f"neighbor_d_t_network_{neighbor}"):
                    self.neighbors_d_t_network.append(
                        d_q_func(
                            (
                                neighbors_degrees[neighbor]
                                + observation_shape[0]
                                - num_actions,
                            ),
                            neighbors_degrees[neighbor],
                            nb_nodes,
                            [1, neighbors_degrees[neighbor]],
                        )
                    )
                    self.neighbors_d_t_network[-1].compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanSquaredError()],
                    )
                self.neighbors_d_t_database.append(DigitalTwinDB(self.d_t_max_time))

    # @tf.function
    def step(self, obs, update_eps=-1, actions_probs=None, stochastic=True):
        """Take an action using the q network. Implements epsilon greedy exploration strategy

        Args:
            obs (list): observation
            update_eps (int, optional): if positive, update the epsilon. Defaults to -1.
            actions_probs (list, optional): Probalities for each action. Defaults to None.
            stochastic (bool, optional): If True, activate epsilon greedy. Defaults to True.

        Returns:
            tuple: action and q values
        """
        q_values = self.q_network(obs)
        # deterministic_actions = tf.argmax(q_values, axis=1)
        deterministic_actions = tf.argmin(q_values, axis=1)
        if stochastic:
            batch_size = tf.shape(obs)[0]
            if actions_probs is None:
                random_actions = tf.random.uniform(
                    tf.stack([batch_size]),
                    minval=0,
                    maxval=self.num_actions,
                    dtype=tf.int64,
                )
            else:
                random_actions = tf.constant(
                    np.random.choice(self.num_actions, p=actions_probs), dtype=tf.int64
                )
            choose_random = (
                tf.random.uniform(
                    tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32
                )
                < self.eps
            )
            output_actions = tf.where(
                choose_random, random_actions, deterministic_actions
            )
        else:
            output_actions = deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)
        return output_actions, q_values

    def save(self, path):
        """Save the model

        Args:
            path (str): path to save the model
        """
        self.q_network.save(path)
        
    # @tf.function()
    def train(self, obs, actions, q_t_selected_targets, importance_weights):
        """
        Train the agent on a batch of sampled experience
            obs: batch of observations
            actions: batch of actions
            q_t_selected_targets: batch of TD targets
            importance_weights: batch of importance weights
            lock: lock to give access to the q network
        """
        with tf.GradientTape() as tape:
            tape.watch(obs)
            q_t = self.q_network(obs)
            q_t_selected = tf.reduce_sum(
                q_t * tf.one_hot(actions, self.num_actions), 1
            )

            td_error = q_t_selected - tf.stop_gradient(q_t_selected_targets)
            errors = huber_loss(td_error)
            # errors = tf.square(td_error)
            weighted_error = tf.reduce_mean(importance_weights * errors)
        grads = tape.gradient(weighted_error, self.q_network.trainable_variables)
        if self.grad_norm_clipping:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
            clipped_grads = grads
        grads_and_vars = zip(grads, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return td_error

    # tf.function(autograph=False)
    def update_target(self):
        """Update the target q network"""
        # pass
        q_vars = self.q_network.trainable_variables
        target_q_vars = self.target_q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)

    def get_target_value(self, rewards, obs1, dones, filtered_indices):
        """Get the target value for the q network

        Arguments:
            rewards {tf.Tensor} -- rewards
            obs1 {tf.Tensor} -- next observations
            dones {tf.Tensor} -- done flags
            filtered_indices {tf.Tensor} -- indices of the actions to be considered
        """
        q_tp1 = tf.gather(self.q_network(obs1), filtered_indices, axis=1)

        #   if self.double_q:
        #     q_tp1_using_online_net = tf.gather(self.q_network(obs1), filtered_indices, axis=1)
        #     #q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
        #     q_tp1_best_using_online_net = tf.argmin(q_tp1_using_online_net, 1)
        #     q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, len(filtered_indices), dtype=tf.float32), 1)
        #   else:
        # q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best = tf.reduce_min(q_tp1, 1)

        dones = tf.cast(dones, q_tp1_best.dtype)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best

        q_t_selected_targets = rewards + self.gamma * q_tp1_best_masked

        return q_t_selected_targets

    def get_neighbor_target_value(
        self, neighbor_idx, rewards, obs1, dones, filtered_indices
    ):
        """Return the target values using the neighbor stored target q network.

        Args:
            neighbor_idx (int): neighbor index
            rewards (tf or np array): values of the reward
            obs1 (tf or np array): the states at the neighbor (s')
            dones (list of bool): if the neighbor is the destination
            filtered_indices (list): indices to filter from s'
        Returns:
            tf tensor: the target values
        """
        # print(type(self.neighbors_target_q_networks))
        # print("get_target_value", neighbor_idx ,  rewards.shape, obs1.shape, dones.shape, self.neighbors_target_q_networks[neighbor_idx].input_shape)
        q_tp1 = tf.gather(
            self.neighbors_target_q_networks[neighbor_idx](obs1),
            filtered_indices,
            axis=1,
        )
        # q_tp1 = self.neighbors_target_q_networks[neighbor_idx](obs1)

        q_tp1_best = tf.reduce_min(q_tp1, 1)

        dones = tf.cast(dones, q_tp1_best.dtype)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best

        q_t_selected_targets = rewards + self.gamma * q_tp1_best_masked

        return q_t_selected_targets

    def sync_neighbor_target_q_network(self, neighbor_idx, with_temp=False):
        """Copy upcoming target nn into neighbor target q network attribute

        Args:
            agent_nn (DQN agent): agent containing the neural network to copy
            neighbor_idx (int): neighbor index
            with_temp (bool, optional): if True, copy the temp upcoming target nn. Defaults to False.
        """
        if with_temp:
            q_vars = self.neighbors_target_temp_upcoming_q_networks[
                neighbor_idx
            ].trainable_variables
            target_q_vars = self.neighbors_target_q_networks[
                neighbor_idx
            ].trainable_variables
            for var, var_target in zip(q_vars, target_q_vars):
                var_target.assign(var)
        else:
            q_vars = self.neighbors_target_upcoming_q_networks[
                neighbor_idx
            ].trainable_variables
            target_q_vars = self.neighbors_target_q_networks[
                neighbor_idx
            ].trainable_variables
            for var, var_target in zip(q_vars, target_q_vars):
                var_target.assign(var)

    def get_neighbor_d_t_value(
        self, neighbor_idx, rewards, obs1, dones, filtered_indices
    ):
        """Return the target values using the digital twin of the neighbor target q network.

        Args:
            neighbor_idx (int): neighbor index
            rewards (tf or np array): values of the reward
            obs1 (tf or np array): the states at the neighbor (s')
            dones (list of bool): if the neighbor is the destination
            filtered_indices (list): indices to filter from s'
        Returns:
            tf tensor: the target values
        """
        q_tp1 = tf.gather(
            self.neighbors_d_t_network[neighbor_idx](obs1), filtered_indices, axis=1
        )

        q_tp1_best = tf.reduce_min(q_tp1, 1)

        dones = tf.cast(dones, q_tp1_best.dtype)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best

        q_t_selected_targets = rewards + self.gamma * q_tp1_best_masked

        return q_t_selected_targets

    def sync_neighbor_upcoming_target_q_network(self, agent_nn, neighbor_idx):
        """Copy nn network into neighbor upcoming target q network attribute

        Args:
            agent_nn (DQN agent): agent containing the neural network to copy
            neighbor_idx (int): neighbor index
        """
        q_vars = agent_nn.q_network.trainable_variables
        if self.neighbors_target_upcoming_q_networks[neighbor_idx] == []:
            with tf.name_scope(f"neighbors_target_upcoming_q_network_{neighbor_idx}"):
                self.neighbors_target_upcoming_q_networks[neighbor_idx] = self.q_func(
                    agent_nn.observation_shape,
                    agent_nn.num_actions,
                    self.nb_nodes,
                    agent_nn.input_size_splits,
                )
        # copy old weights of upcoming into temp upcoming
        target_q_vars = self.neighbors_target_temp_upcoming_q_networks[
            neighbor_idx
        ].trainable_variables
        for var, var_target in zip(
            self.neighbors_target_upcoming_q_networks[neighbor_idx].trainable_variables,
            target_q_vars,
        ):
            var_target.assign(var)

        ## copy weights of agent nn into upcoming
        target_q_vars = self.neighbors_target_upcoming_q_networks[
            neighbor_idx
        ].trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)
