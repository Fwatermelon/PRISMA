#!/usr/bin python3
# -*- coding: utf-8 -*-
""" -----Main file for the PRISMA project----- """


__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, TDS Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

import os
import shlex
import signal
import datetime
import subprocess
import threading
from time import sleep, time
import viztracer
from source.auxilliaries.argument_parser import parse_arguments
from source.auxilliaries.run_ns3 import run_ns3
from source.controller.agents.base_dqn_routing_logger import BaseDqnRoutingLogger
from source.controller.agents.dqn_routing_logit_sharing import LogitSharingAgent
from source.controller.agents.dqn_routing_model_sharing import ModelSharingAgent
from source.controller.agents.dqn_routing_value_sharing import ValueSharingAgent
from source.controller.agents.classical_routing_methods import ShortestPathAgent, OracleAgent
from source.controller.agents.dqn_routing_centralized import CentralizedMADRLAgent
from source.controller.base_logger import BaseLogger
from source.auxilliaries.utils import allocate_on_gpu, fix_seed


def main() -> (int, int):
    """Main entry point for the script.
    It first parses the arguments, then runs the ns3 simulator and the agents threads for training or testing.

    Returns:
        (ns3_proc_id, tensorboard_process): process ids of the ns3 simulator and the tensorboard server
    """
    ### Allocate GPU memory as needed
    allocate_on_gpu()

    ### Get the arguments from the parser
    params = parse_arguments()

    ### fix the seed
    fix_seed(params["seed"])

    ### check the agent type
    if params["agent_type"] == "shortest_path":
        Agent = ShortestPathAgent
        Logger = BaseLogger
    elif params["agent_type"] == "oracle_routing":
        Agent = OracleAgent
        Logger = BaseLogger
    elif params["agent_type"] == "dqn_model_sharing":
        Agent = ModelSharingAgent
        Logger = BaseDqnRoutingLogger
    elif params["agent_type"] == "dqn_value_sharing":
        Agent = ValueSharingAgent
        Logger = BaseDqnRoutingLogger
    elif params["agent_type"] == "dqn_logit_sharing":
        Agent = LogitSharingAgent
        Logger = BaseDqnRoutingLogger
    elif params["agent_type"] == "madrl_centralized":
        Agent = CentralizedMADRLAgent
        Logger = BaseDqnRoutingLogger
    else:
        raise ValueError(
            f"Unknown agent type : {params['agent_type']}. Please choose one of the following : shortest_path, oracle_routing, dqn_model_sharing, dqn_value_sharing, dqn_logit_sharing"
        )

    ### fill model version
    # TODO: fix testing different model versions
    # if "dqn_buffer" not in params["agent_type"] or params["train"] == 1:
    #     params["model_version"] = ""
    # else:
    #     params["model_version"] = params["load_path"].split("/")[-1]
    ### initialize the logger
    logger_object = Logger(params)

    ### check if the session name is already used
    # TODO: add overwrite option
    if logger_object.check_session_exists():
        print(
            f"The session {params['session_name']} already exists. No simulation will be run."
        )
        return None, None

    ### setup the agents (fix the static variables)
    Agent.init_static_vars(params, logger_object)

    ### start the profiler
    if params["profile_session"]:
        tracer = viztracer.VizTracer(
            tracer_entries=5000000,
            min_duration=100,
            max_stack_depth=20,
            output_file=f"{params['logs_parent_folder'].rstrip('/')}/{params['session_name']}/viztracer.json",
        )
        tracer.start()

    ## Run tensorboard server in a separate process
    tensorboard_process = None
    # TODO: start tensorboard without subprocess
    if params["start_tensorboard"]:
        args = [
            "python3",
            "-m",
            "tensorboard.main",
            f"--logdir={shlex.quote(logger_object.logs_folder)}",
            f'--port={params["tensorboard_port"]}',
            "--bind_all",
        ]
        tensorboard_process = subprocess.Popen(args)
        print(f"Tensorboard server started with process ID {tensorboard_process}")

    nodes_objects = []
    # snapshot_index = 1
    ## run ns3 simulator
    for episode in range(params["nb_episodes"]):
        print("running ns-3")
        ns3_proc_id = run_ns3(params)
        Agent.reset()
        logger_object.logging["episode_index"] = episode
        ## run the agents threads
        for index in params["G"].nodes():
            print("Starting agent", index)
            if episode == 0:
                ## create the agent class instance
                node_object = Agent(index)
                nodes_objects.append(node_object)
            else:
                nodes_objects[index].reset_node()
            ## start the agent forwarder thread
            th = threading.Thread(target=nodes_objects[index].run, args=())
            th.start()

        sleep(1)

        ## wait until simulation completes
        while threading.active_count() > params["nb_nodes"]:
            sleep(5)
            # TODO: save the models after each episode
            # if params["train"] == 1:
            # stats_writer_train(
            #     summary_writer_session,
            #     summary_writer_nb_arrived_pkts,
            #     summary_writer_nb_lost_pkts,
            #     summary_writer_nb_new_pkts,
            #     Agent,
            # )
            # ## check if it is time to save a snapshot of the models
            # if (
            #     (Agent.base_curr_time + Agent.curr_time)
            #     > (snapshot_index * params["snapshot_interval"])
            # ) and params["snapshot_interval"] > 0:
            #     print(
            #         f"Saving model at time {Agent.curr_time} with index {snapshot_index}"
            #     )
            #     save_all_models(
            #         Agent.agents,
            #         params["G"].nodes(),
            #         params["session_name"],
            #         snapshot_index,
            #         episode,
            #         root=params["saved_models_path"],
            #         snapshot=True,
            #     )
            #     snapshot_index += 1
        sleep(5)
        logger_object.print_summary()
        if params["train"] == 1 and node_object.logger.logging["curr_time"] > (params["episode_duration"]*params["nb_episodes"]*0.95):
            print("Saving models", node_object.logger.logging["curr_time"], (params["episode_duration"]*params["nb_episodes"]*0.95))
            for node_object in nodes_objects:
                node_object.save_model()
    # if params["train"] == 0:
    logger_object.stats_final()

    ## save the replay buffers
    # if params["train"] == 1:
    #     for idxx, rb in enumerate(Agent.replay_buffer):
    #         if not os.path.exists(params["logs_folder"] + "/replay_buffers"):
    #             os.mkdir(path=params["logs_folder"] + "/replay_buffers")
    #         rb.save(params["logs_folder"] + "/replay_buffers/" + str(idxx) + ".pkl")

    ## save the profiler results
    if params["profile_session"]:
        tracer.stop()
        tracer.save()


    return (ns3_proc_id, tensorboard_process)


if __name__ == "__main__":
    ## create a process group
    import traceback

    ns3_pid, tb_process = None, None
    # os.setpgrp()
    try:
        print("starting process group")
        start_time = time()
        ns3_pid, tb_process = main()
        print("Elapsed time = ", str(datetime.timedelta(seconds=time() - start_time)))
    except Exception:
        traceback.print_exc()
        # write the error in the log file
        with open("examples/error.log", "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
    finally:
        print("kill process group")
        if ns3_pid:
            os.system(command=f"kill -9 {ns3_pid}")
        if tb_process:
            os.system(command=f"kill -9 {tb_process}")
        import sys
        sys.exit(0)
        # os.killpg(0, signal.SIGKILL)
