""" This file contains the base class for the logger. It is used to log the
    simulation results in using tensorboard.
"""
import os
import time
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from source.auxilliaries.utils import convert_tb_data

class BaseLogger:
    """ Base class for the logger. It is used to log the simulation results in using tensorboard.
    """
    def __init__(self, params_dict):
        """ Initialize the logger

        Args:
            params_dict (dict): dictionary containing the simulation parameters
        """
        ## store the parameters
        print("Initializing the logger")
        self.load_factor = params_dict["load_factor"]
        self.logging_timestep = params_dict["logging_timestep"]
        ## Add session name if not specified
        if params_dict["session_name"] is None:
            params_dict["session_name"] = datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
        self.session_name = params_dict["session_name"]
        self.logs_parent_folder = params_dict["logs_parent_folder"]
        ### create the logs folder
        self.logs_folder = (
            self.logs_parent_folder + "/" + self.session_name
        )
        self.global_stats_path = f"{self.logs_folder}/stats"
        self.nb_arrived_pkts_path = f"{self.logs_folder}/nb_arrived_pkts"
        self.nb_new_pkts_path = f"{self.logs_folder}/nb_new_pkts"
        self.nb_lost_pkts_path = f"{self.logs_folder}/nb_lost_pkts"
        self.nb_lost_pkts_detected_path = f"{self.logs_folder}/nb_lost_pkts_detected"
        self.nb_nodes = params_dict["nb_nodes"]
        if not os.path.exists(self.logs_folder):
            os.makedirs(self.logs_folder)
        self.tests_path = f"{self.logs_folder}/test_results"
        # logging attributes (used to collect and log data from ns3 and produce statics)
        self.logging = {
            "curr_time": 0,  # current time for the simulation
            "episode_index": 0,  # current episode
            "episode_time": 0,  # current time for the episode
            "nodes_nb_iterations": [0 for _ in range(params_dict["nb_nodes"])],  # number of iterations for each node
            "total_nb_iterations": 0,  # total number of iterations
            "total_hops": 0,  # total number of hops
            "total_new_rcv_pkts": 0,  # total number of arrived packets
            "total_data_size": 0,  # total data size
            "total_rewards_with_loss": 0,  # total rewards with loss
            "total_arrived_pkts": 0,  # total number of arrived packets
            "total_e2e_delay_measured": 0,  # total end to end delay measured
            "total_e2e_delay_computed": 0,  # total end to end delay computed
            "notified_lost_pkts": 0,  # total number of notified lost packets
            "lost_pkts_detected": 0,  # total number of lost packets detected
            "nb_memory_update_pkts": 0,  # total number of replay memory update packets received
            "replay_memory_update_overhead_counter": 0,  # total size of replay memory update packets received
            "sim_injected_packets": 0,  # number of injected packets for overlay from ns3
            "sim_global_injected_packets": 0,  # number of injected packets globally from ns3
            "sim_dropped_packets": 0,  # number of dropped packets from ns3
            "sim_rejected_packets": 0,  # number of rejected packets from ns3
            "sim_global_dropped_packets": 0,  # number of dropped packets globally from ns3
            "sim_global_rejected_packets": 0,  # number of rejected packets globally from ns3
            "sim_delivered_packets": 0,  # number of delivered packets from ns3
            "sim_global_delivered_packets": 0,  # number of delivered packets globally from ns3
            "sim_buffered_packets": 0,  # number of buffered packets from ns3
            "sim_signaling_overhead": 0,  # signaling overhead from ns3
            "sim_global_buffered_packets": 0,  # number of buffered packets globally from ns3
            "sim_avg_e2e_delay": 0.0,  # average end to end delay from ns3
            "sim_sum_e2e_delay": 0.0,  # sum of end to end delay from ns3
            "sim_cost": 0.0,  # cost from ns3
            "sim_global_avg_e2e_delay": 0.0,  # average end to end delay globally from ns3
            "sim_global_cost": 0.0,  # cost globally from ns3
            "mv_avg_e2e_delay": 0.0,  # moving average of end to end delay
            "mv_avg_loss": 0.0,  # moving average of loss
            "mv_nb_hops": 0.0,  # moving average of number of hops
            "mv_overhead": 0.0,  # moving average of overhead
            "mv_injected_pkts": 0.0,  # moving average of injected packets
        }
        ### define the tensorboard writers
        self.summary_writer_parent = None
        self.summary_writer_session = None
        self.summary_writer_nb_arrived_pkts = None
        self.summary_writer_nb_new_pkts = None
        self.summary_writer_nb_lost_pkts = None
        self.summary_writer_nb_lost_pkts_detected = None
        self.logging["start_time"] = time.time()
        self.logging["last_log_time"] = 0
        self.create_tensorboard_writer()
        self.reset()
        
    def reset(self):
        """ Reset the logger attributes when a new episode starts
        """
        self.e2e_delay_per_flow = [[ [] for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.mv_avg_e2e_delay = pd.Series(dtype=float)
        self.mv_avg_loss = pd.Series(dtype=float)
        self.mv_nb_hops = pd.Series(dtype=float)
        self.mv_overhead = pd.Series(dtype=float)
        self.mv_injected_pkts = pd.Series(dtype=float)
        self.nb_hops_per_flow = [[ [] for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.paths_per_flow = [[ set() for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.jitter_per_flow = [[ [] for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.lost_packets_per_flow = [[0 for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        
    def create_tensorboard_writer(self):
        """Create the tensorboard writer
        """
        ## delete the previous logs
        if os.path.exists(self.logs_folder):
            os.system(f"rm -rf {self.logs_folder}")
        ### create the tensorboard writer when training
        self.summary_writer_parent = tf.summary.create_file_writer(
            logdir=self.logs_folder
        )
        self.summary_writer_session = tf.summary.create_file_writer(
            logdir=self.global_stats_path
        )
        self.summary_writer_nb_arrived_pkts = tf.summary.create_file_writer(
            logdir=self.nb_arrived_pkts_path
        )
        self.summary_writer_nb_new_pkts = tf.summary.create_file_writer(
            logdir=self.nb_new_pkts_path
        )
        self.summary_writer_nb_lost_pkts = tf.summary.create_file_writer(
            logdir=self.nb_lost_pkts_path
        )
        self.summary_writer_nb_lost_pkts_detected = tf.summary.create_file_writer(
            logdir=self.nb_lost_pkts_detected_path
        )


    def log_stats(self):
        """Log the stats to tensorboard
        """
        nb_iteration = self.logging["total_nb_iterations"]
        curr_time = int((self.logging["curr_time"]) * 1e6)
        ## write the global stats
        if self.logging["sim_injected_packets"] > 0:
            loss_ratio = (
                self.logging["sim_dropped_packets"]
                / self.logging["sim_injected_packets"]
            )
        else:
            loss_ratio = -1
        if self.logging["sim_delivered_packets"] > 0:
            avg_delay_overlay = self.logging["sim_avg_e2e_delay"]
            avg_delay_global = self.logging["sim_global_avg_e2e_delay"]
            avg_cost_overlay = self.logging["sim_cost"]
            avg_cost_global = self.logging["sim_global_cost"]
            avg_hops = (
                self.logging["total_hops"] / self.logging["sim_delivered_packets"]
            )
        else:
            avg_delay_overlay = -1
            avg_delay_global = -1
            avg_cost_overlay = -1
            avg_cost_global = -1
            avg_hops = -1
            
        with self.summary_writer_session.as_default():
            ## total rewards
            tf.summary.scalar(
                "total_e2e_delay_computed_over_time",
                self.logging["total_e2e_delay_computed"],
                step=curr_time,
            )
            tf.summary.scalar(
                "total_e2e_delay_over_time",
                self.logging["total_e2e_delay_measured"],
                step=curr_time,
            )
            tf.summary.scalar(
                "total_rewards_with_loss_over_time",
                self.logging["total_rewards_with_loss"],
                step=curr_time,
            )
            ## loss ratio
            tf.summary.scalar(
                "loss_ratio_over_time",
                loss_ratio,
                step=curr_time,
            )
            ## total hops and avg hops
            tf.summary.scalar(
                "total_hops_over_time",
                self.logging["total_hops"],
                step=curr_time,
            )
            tf.summary.scalar(
                "avg_hops_over_time",
                avg_hops,
                step=curr_time,
            )
            # tf.summary.scalar(
            #     "ma_avg_hops_over_time",
            #     np.array(self.logging["nb_hops"]).mean(),
            #     step=curr_time,
            # )
            ## buffers occupation
            tf.summary.scalar(
                "nb_buffered_pkts_over_time",
                self.logging["sim_buffered_packets"],
                step=curr_time,
            )
            ## signalling overhead
            tf.summary.scalar('sim_signaling_overhead_per_time', self.logging["sim_signaling_overhead"], step=curr_time)

            ### avg cost and avg delay
            # tf.summary.scalar(
            #     "avg_cost_over_iterations",
            #     avg_cost,
            #     step=nb_iteration,
            # )
            tf.summary.scalar(
                "sim_cost_over_time",
                avg_cost_overlay,
                step=curr_time,
            )
            tf.summary.scalar(
                "sim_global_cost_over_time",
                avg_cost_global,
                step=curr_time,
            )
            tf.summary.scalar(
                "sim_e2e_delay_over_time",
                avg_delay_overlay,
                step=curr_time,
            )
            tf.summary.scalar(
                "sim_global_e2e_delay_over_time",
                avg_delay_global,
                step=curr_time,
            )
            
            # slice the series to keep only the last logging_timestep
            self.mv_avg_e2e_delay = self.mv_avg_e2e_delay[self.mv_avg_e2e_delay.index>= self.logging["curr_time"]-self.logging_timestep]
            self.mv_avg_loss = self.mv_avg_loss[self.mv_avg_loss.index>= self.logging["curr_time"]-self.logging_timestep]
            self.mv_nb_hops = self.mv_nb_hops[self.mv_nb_hops.index>= self.logging["curr_time"]-self.logging_timestep]
            self.mv_overhead = self.mv_overhead[self.mv_overhead.index>= self.logging["curr_time"]-self.logging_timestep]
            self.mv_injected_pkts = self.mv_injected_pkts[self.mv_injected_pkts.index>= self.logging["curr_time"]-self.logging_timestep]
            if len(self.mv_avg_e2e_delay) > 0:
                tf.summary.scalar(
                    "mv_avg_e2e_delay_over_time",
                    self.mv_avg_e2e_delay.mean(),
                    step=curr_time,
                )
                tf.summary.scalar(
                    "mv_avg_loss_over_time",
                    self.mv_avg_loss.sum()/len(self.mv_avg_e2e_delay),
                    step=curr_time,
                )
            if len(self.mv_nb_hops) > 0:
                tf.summary.scalar(
                    "mv_nb_hops_over_time",
                    self.mv_nb_hops.mean(),
                    step=curr_time,
                )
            if len(self.mv_injected_pkts) > 0:
                tf.summary.scalar(
                    "mv_overhead_over_time",
                    self.mv_overhead.sum()/self.mv_injected_pkts.sum(),
                    step=curr_time,
                )
            ## simulation time / real time
            tf.summary.scalar(
                "sim_second_per_real_seconds",
                (time.time() - self.logging["start_time"])
                / (self.logging["curr_time"]),
                step=curr_time,
            )
        ### write the arrived pkts
        with self.summary_writer_nb_arrived_pkts.as_default():
            tf.summary.scalar(
                "pkts_over_time",
                self.logging["sim_delivered_packets"],
                step=curr_time,
            )
        ### write the lost pkts
        with self.summary_writer_nb_lost_pkts.as_default():
            tf.summary.scalar(
                "pkts_over_time",
                self.logging["sim_dropped_packets"],
                step=curr_time,
            )
        
        with self.summary_writer_nb_lost_pkts_detected.as_default():
            tf.summary.scalar(
                "pkts_over_time",
                self.logging["lost_pkts_detected"],
                step=curr_time,
            )
        ### write the new pkts
        with self.summary_writer_nb_new_pkts.as_default():
            tf.summary.scalar(
                "pkts_over_time",
                self.logging["sim_injected_packets"],
                step=curr_time,
            )
        # print("logged at time", curr_time, " hops ", avg_hops, " e2e delay ", avg_delay_overlay, " cost ", avg_cost_overlay, " loss ratio ", loss_ratio)
        
    def check_session_exists(self) -> bool:
        """ Check if the session name is already used

        Returns:
            bool: True if the session name is already used, False otherwise
        """

        if os.path.exists(self.tests_path):
            if len(os.listdir(self.tests_path))> 0:
                ## check if the test load factor is already in the tensorboard file
                try:
                    if (int(100 * self.load_factor) in convert_tb_data(self.tests_path)["step"].values):
                            print(f'The test session with load factor {self.load_factor} already exists in the {self.tests_path} folder')
                            return True
                except Exception as e:
                    print(e)
        return False

    def final_stats_writer(self):
        """Write the stats of the session to the logs dir using tensorboard writer during test phase
        """
        # TODO: add the model version
        # model_version = Agent.model_version
        ## create the writer
        summary_writer_results = tf.summary.create_file_writer(
            logdir=f"{self.tests_path}"
        )
        ## store test stats
        with summary_writer_results.as_default():
            tf.summary.scalar(
                "final_global_injected_pkts_per_load",
                self.logging["sim_global_injected_packets"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_overlay_injected_pkts_per_load",
                self.logging["sim_injected_packets"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_global_lost_pkts_per_load",
                self.logging["sim_global_dropped_packets"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_overlay_lost_pkts_per_load",
                self.logging["sim_dropped_packets"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_global_arrived_pkts_per_load",
                self.logging["sim_global_delivered_packets"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_overlay_arrived_pkts_per_load",
                self.logging["sim_delivered_packets"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_global_e2e_delay_per_load",
                self.logging["sim_avg_e2e_delay"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_overlay_e2e_delay_per_load",
                self.logging["sim_global_avg_e2e_delay"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_global_loss_rate_per_load",
                self.logging["sim_global_dropped_packets"]
                / max(1, self.logging["sim_global_injected_packets"]),
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_overlay_loss_rate_per_load",
                self.logging["sim_dropped_packets"] / max(1, self.logging["sim_injected_packets"]),
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_global_cost_per_load",
                self.logging["sim_global_cost"],
                step=int(self.load_factor * 100),
            )
            tf.summary.scalar(
                "final_overlay_cost_per_load", self.logging["sim_cost"], step=int(self.load_factor * 100)
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
                Overlay lost packets python = {self.logging["notified_lost_pkts"]},
                Overlay lost packets detected = {self.logging["lost_pkts_detected"]},
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
                """
                # nbBytesOverlaySignalingForward = {self.logging["sim_bytes_overlay_signaling_forward"]},
                # nbBytesOverlaySignalingBack = {self.logging["sim_bytes_overlay_signaling_back"]},
        )

    def save_flow_data(self,):
        """ Save the lost packets, average end-to-end delay, average hop and jitter per flow (we refer to flow as a source and destination pair) to a file
        """
        with open(f"{self.logs_folder}/lost_packets_per_flow.txt", "w", encoding="utf-8") as f:
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    if i == j:
                        f.write("0 ")
                    else:
                        f.write(f"{self.lost_packets_per_flow[i][j]} ")
                f.write("\n")
        with open(f"{self.logs_folder}/e2e_delay_per_flow.txt", "w", encoding="utf-8") as f:
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    if i == j:
                        f.write("0 ")
                    else:
                        f.write(f"{np.mean(self.e2e_delay_per_flow[i][j])} ")
                f.write("\n")
        with open(f"{self.logs_folder}/nb_hops_per_flow.txt", "w", encoding="utf-8") as f:
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    if i == j:
                        f.write("0 ")
                    else:
                        f.write(f"{np.mean(self.nb_hops_per_flow[i][j])} ")
                f.write("\n")
        # save the list of sets object
        with open(f"{self.logs_folder}/paths_per_flow.txt", "w", encoding="utf-8") as f:
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    f.write(f"{i} {j} ")
                    if i == j:
                        f.write("0 ")
                    else:
                        f.write(f"{self.paths_per_flow[i][j]} ")
                    f.write("\n")
                f.write("\n")
                
        with open(f"{self.logs_folder}/jitter_per_flow.txt", "w", encoding="utf-8") as f:
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    if i == j:
                        f.write("0 ")
                    else:
                        f.write(f"{np.std(self.e2e_delay_per_flow[i][j])} ")
                f.write("\n")

    def reset_moving_stats(self):
        """Reset the moving stats
        """
        # print("Resetting the moving stats at time", self.logging["curr_time"])
        # print("nb_hops_per_flow", self.nb_hops_per_flow)
        # print("paths_per_flow", self.paths_per_flow)
        # print("lost_packets_per_flow", self.lost_packets_per_flow)
        # print("e2e_delay_per_flow", self.e2e_delay_per_flow)
        
        self.nb_hops_per_flow = [[ [] for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.paths_per_flow = [[ set() for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.jitter_per_flow = [[ [] for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.lost_packets_per_flow = [[0 for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        self.e2e_delay_per_flow = [[ [] for _ in range(self.nb_nodes)] for _ in range(self.nb_nodes)]
        
        

    def stats_final(self):
        """Log the final stats to tensorboard
        """
        # self.final_stats_writer()
        self.save_flow_data()