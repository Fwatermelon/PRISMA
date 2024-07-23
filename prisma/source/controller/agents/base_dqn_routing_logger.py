""" This file contains the base class for the logger. It is used to log the
    simulation results in using tensorboard.
"""
import copy
import os
import time

import tensorflow as tf
from source.controller.base_logger import BaseLogger
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.custom_scalar import summary as cs_summary
from tensorboard.plugins.hparams import api as hp


class BaseDqnRoutingLogger(BaseLogger):
    """Base class for the logger specific to the DQN routing agent. It is used to log the simulation results in using tensorboard. 
    """

    def __init__(self, params_dict):
        """Initialize the logger
        Args:
            params_dict (dict): dictionary containing the simulation parameters
        """
        super().__init__(params_dict)
        ## store the parameters
        self.train = params_dict["train"]
        if self.train == 1:
            self.saved_models_path = params_dict["saved_models_path"] + "/" + self.session_name
        self.target_update_period = params_dict["target_update_period"]
        
        ## add some params for tensorboard writer
        if self.train == 1:
            self.td_errors_path = f"{self.logs_folder}/td_errors"
            self.rb_buffer_lengths_path = f"{self.logs_folder}/replay_buffer_length"
            self.lambda_coefs_path = f"{self.logs_folder}/lambda_coefs"
            self.summary_writer_td_errors = None
            self.summary_writer_rb_buffer_lengths = None
            self.summary_writer_lambda_coefs = None
            ### add new logging for tracking the exploration and training
            self.logging["epsilon"] = [0 for _ in range(params_dict["nb_nodes"])]
            self.logging["nb_target_update_pkts"] = 0
            self.logging["nb_memory_update_pkts"] = 0
            self.logging["target_update_overhead_size"] = 0
            self.logging["replay_memory_update_overhead_size"] = 0
            self.logging["td_errors"] = [0 for _ in range(params_dict["nb_nodes"])]
            self.logging["replay_buffer_lengths"] = [0 for _ in range(params_dict["nb_nodes"])]
            self.logging["gap"] = 0
            self.logging["sync_counter"] = 0
            self.create_tensorboard_writer_train(params_dict)

    def create_tensorboard_writer_train(self, params_dict: dict):
        """Create the tensorboard writer
        
        Args:
            params_dict (dict): dictionary containing the simulation parameters
        """
        self.summary_writer_exploration = [tf.summary.create_file_writer(logdir=f"{self.logs_folder}/exploration/node_{i}") for i in range(self.nb_nodes)]
        self.summary_writer_td_errors = [tf.summary.create_file_writer(logdir=f"{self.td_errors_path}/node_{i}") for i in range(self.nb_nodes)]
        self.summary_writer_rb_buffer_lengths = [tf.summary.create_file_writer(logdir=f"{self.rb_buffer_lengths_path}/node_{i}") for i in range(self.nb_nodes)]
        self.summary_writer_lambda_coefs = [[tf.summary.create_file_writer(logdir=f"{self.lambda_coefs_path}/node_{i}/lambda_{j}") for j in range(self.nb_nodes-1)] for i in range(self.nb_nodes)]
        

        ## write the session info (parameters)
        with tf.summary.create_file_writer(logdir=self.logs_folder).as_default():
            ## Adapt the dict to the hparams api
            dict_to_store = copy.deepcopy(params_dict)
            ## convert non serializable objects to string
            for key, value in dict_to_store.items():
                if not isinstance(value, (int, float, str, bool)):
                    dict_to_store[key] = str(value)
            hp.hparams(dict_to_store)  # record the values used in this trial

        ## Define the custom categories in tensorboard
        with self.summary_writer_parent.as_default():
            tf.summary.experimental.write_raw_pb(
                self.custom_plots().SerializeToString(), step=0
            )

    def check_session_exists(self) -> bool:
        """Check if the session name is already used

        Returns:
            bool: True if the session name is already used, False otherwise
        """
        # TODO: add model versions
        if (
            self.train == 1 and self.saved_models_path is not None
        ):  # if we are training the model, we check the saved_models_path
            print("Checking if the session already exists", self.saved_models_path)
            # pathlib.Path(self.saved_models_path).mkdir(parents=True, exist_ok=True)
            if os.path.exists(self.saved_models_path):
                if len(os.listdir(self.saved_models_path)) > 0:
                    print(
                        f"The session {self.session_name} already exists in : {self.saved_models_path}"
                    )
                    return True
            return False
        else:  # if we are testing the model, we check the logs folder
            return super().check_session_exists()

    def stats_writer_train(self):
        """Write the stats of the session to the logs dir using tensorboard writer during the training"""
        nb_iteration = self.logging["total_nb_iterations"]
        curr_time = int((self.logging["curr_time"]) * 1e6)

        with self.summary_writer_session.as_default():
            ## exploration value
            for index in range(self.nb_nodes):
                with self.summary_writer_exploration[index].as_default():
                    tf.summary.scalar(
                        f"exploration_value_over_time",
                        self.logging["epsilon"][index],
                        step=curr_time,
                    )
            ## sync counter divided by the number of nodes* total syncs (to have the number of syncs per target update period per node)
            tf.summary.scalar(
                "effective_sync_periode",
                self.nb_nodes * self.logging["curr_time"]/( max(1, self.logging["sync_counter"])),
                step=curr_time,
            )
            ## write gap
            tf.summary.scalar(
                "gap_over_time",
                self.logging["gap"],
                step=curr_time,
            )
            ## iteration over time
            tf.summary.scalar(
                "iteration_over_time",
                nb_iteration,
                step=curr_time,
            )
            # TODO: add the signalling bytes
            tf.summary.scalar('replay_memory_update_overhead_size_per_time', self.logging["replay_memory_update_overhead_size"], step=curr_time)
            tf.summary.scalar('target_update_overhead_size_per_time', self.logging["target_update_overhead_size"], step=curr_time)
            tf.summary.scalar('nb_memory_update_pkts_per_sec', self.logging["nb_memory_update_pkts"]/max(1, self.logging["curr_time"]), step=curr_time)
            # tf.summary.scalar('overlay_big_signalling_bytes', self.logging["sim_bytes_big_signaling"], step=int((self.logging["curr_time"])*1e6))
            # tf.summary.scalar('overlay_small_signalling_bytes', self.logging["sim_bytes_small_signaling"], step=int((self.logging["curr_time"])*1e6))
            # tf.summary.scalar('overlay_ping_signalling_bytes', self.logging["sim_bytes_overlay_signaling_back"] + self.logging["sim_bytes_overlay_signaling_forward"], step=int((self.logging["curr_time"])*1e6))

        ### write the td errors
        for index in range(self.nb_nodes):
            with self.summary_writer_td_errors[index].as_default():
                tf.summary.scalar(
                    "td_error_over_time",
                    self.logging["td_errors"][index],
                    step=curr_time,
                )
        ### write the replay buffer length
        for index in range(self.nb_nodes):
            with self.summary_writer_rb_buffer_lengths[index].as_default():
                tf.summary.scalar(
                    "replay_buffer_length_over_time",
                    self.logging["replay_buffer_lengths"][index],
                    step=curr_time,
                )

    def log_lambda_coefs(self, node_id, port_id, lambda_coef):
        """Log the lambda coefficients
        Args:
            node_id (int): the node id
            port_id (int): the port id
            lambda_coef (float): the lambda coefficient
        """
        ### write the lambda coefs
        with self.summary_writer_lambda_coefs[node_id][port_id].as_default():
            tf.summary.scalar(
                "lambda_coef_over_time",
                lambda_coef,
                step=int((self.logging["curr_time"]) * 1e6)
            )

    def print_summary(self):
        """Print the summary of the session
        """
        print(
            f""" Summary of the Episode {self.logging["episode_index"]}:
                Simulation time = {self.logging["curr_time"]},
                Total Iterations = {self.logging["total_nb_iterations"]},
                Overlay Total injected packets = {self.logging["sim_injected_packets"]},
                Global Total injected packets = {self.logging["sim_global_injected_packets"]},
                Overlay arrived packets = {self.logging["sim_delivered_packets"]},
                Global arrived packets = {self.logging["sim_global_delivered_packets"]},
                Overlay lost packets = {self.logging["sim_dropped_packets"]},
                Overlay rejected packets = {self.logging["sim_rejected_packets"]},
                Global lost packets = {self.logging["sim_global_dropped_packets"]},
                Global lost packets python = {self.logging["notified_lost_pkts"]},
                Global rejected packets = {self.logging["sim_global_rejected_packets"]},
                Overlay buffered packets = {self.logging["sim_buffered_packets"]},
                Global buffered packets = {self.logging["sim_global_buffered_packets"]},
                Overlay lost ratio = {self.logging["sim_dropped_packets"]/max(self.logging["sim_injected_packets"], 1.0)},
                Global lost ratio = {self.logging["sim_global_dropped_packets"]/max(self.logging["sim_global_injected_packets"],1.0)},
                Overlay delivered ratio = {self.logging["sim_delivered_packets"]/max(self.logging["sim_injected_packets"], 1.0)},
                Global delivered ratio = {self.logging["sim_global_delivered_packets"]/max(self.logging["sim_global_injected_packets"],1.0)},
                Overlay e2e delay = {self.logging["sim_avg_e2e_delay"]},
                Global e2e delay = {self.logging["sim_global_avg_e2e_delay"]},
                Overlay Cost = {self.logging["sim_cost"]},
                Global Cost = {self.logging["sim_global_cost"]},
                Hops = {self.logging["total_hops"]/max(self.logging["sim_delivered_packets"], 1.0)},
                OverheadRatio = {self.logging["sim_signaling_overhead"]}
                Total Rewards = {self.logging["total_rewards_with_loss"]},
                Total Data Size = {self.logging["total_data_size"]},
                Total New Received Packets = {self.logging["total_new_rcv_pkts"]},
                Total Arrived Packets = {self.logging["total_arrived_pkts"]},
                Total E2E Delay Measured = {self.logging["total_e2e_delay_measured"]},
                Total E2E Delay Computed = {self.logging["total_e2e_delay_computed"]},
                Total Memory Update Packets = {self.logging["nb_memory_update_pkts"]},
                Total Memory Update Overhead = {self.logging["replay_memory_update_overhead_size"]},
                Total Target update Overhead = {self.logging["target_update_overhead_size"]},
                Total Target update Packets = {self.logging["nb_target_update_pkts"]},
                """
                # nbBytesOverlaySignalingForward = {self.logging["sim_bytes_overlay_signaling_forward"]},
                # nbBytesOverlaySignalingBack = {self.logging["sim_bytes_overlay_signaling_back"]},
        )

    def custom_plots(self):
        """define the costum plots for tensorboard
        The user can define the plots he wants to see in tensorboard custom plots, by adding an item to the list of the category
        """
        cs = cs_summary.pb(
            layout_pb2.Layout(
                category=[
                    layout_pb2.Category(
                        title="Main evaluation metrics",
                        chart=[
                            layout_pb2.Chart(
                                title="Avg Delay per arrived pkts (Overlay)",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"sim_e2e_delay_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="Avg Delay per arrived pkts (Global)",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"sim_global_e2e_delay_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="Avg Cost (Overlay)",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"sim_cost_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="Avg Cost (Global)",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"sim_global_cost_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="Loss Ratio (Overlay)",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"loss_ratio_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="Loss Ratio (global)",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"loss_ratio_over_time"]
                                ),
                            ),
                        ],
                    ),
                    layout_pb2.Category(
                        title="Global info about the env",
                        chart=[
                            layout_pb2.Chart(
                                title="Average hops over time",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"avg_hops_over_time"])),
                            layout_pb2.Chart(
                                title="Overhead ratio",
                                multiline=layout_pb2.MultilineChartContent(tag=[r"sim_signaling_overhead"])),
                            # layout_pb2.Chart(
                            #     title="Buffers occupation",
                            #     multiline=layout_pb2.MultilineChartContent(tag=[r"nb_buffered_pkts_over_time"])),
                            # layout_pb2.Chart(
                            #     title="new pkts vs lost pkts vs arrived pkts",
                            #     multiline=layout_pb2.MultilineChartContent(tag=[r"(total_new_rcv_pkts_over_time | total_lost_pkts_over_time | total_arrived_pkts_over_time)"])),
                        ]),
                    layout_pb2.Category(
                        title="Training metrics",
                        chart=[
                            layout_pb2.Chart(
                                title="Td error",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"td_error_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="exploration value",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"exploration_value_over_time"]
                                ),
                            ),
                            layout_pb2.Chart(
                                title="replay buffers length",
                                multiline=layout_pb2.MultilineChartContent(
                                    tag=[r"replay_buffer_length_over_time"]
                                ),
                            ),
                        ],
                    ),
                ]
            )
        )
        return cs
