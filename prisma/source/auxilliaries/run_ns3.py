__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__version__ = "0.1.0"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"

import os
import shlex
import subprocess
import pathlib


def run_ns3(params, configure=True):
    """
    Run the ns3 simulator
    Args:
        params(dict): parameter dict
        configure(bool): if True, run ns3 configure
    Returns:
        proc: process id of the ns3 simulator
    """

    ## check if ns3-gym is in the folder
    if "waf" not in os.listdir(params["ns3_sim_path"]):
        raise Exception(
            f'Unable to locate ns3-gym in the folder : {params["ns3_sim_path"]}'
        )

    ## store current folder path
    current_folder_path = os.getcwd()

    ## Copy prisma into ns-3 folder
    # TODO: make secure copy
    print("Copying prisma into ns-3 folder", "...", f"{pathlib.Path('./source/simulator/ns3/').resolve()}{'/*'}", f'{pathlib.Path(params["ns3_sim_path"]).resolve()}/scratch/prisma/.')
    os.system(f"rsync -r {pathlib.Path('./source/simulator/ns3/').resolve()}{'/*'} {pathlib.Path(params['ns3_sim_path']).resolve()}/scratch/prisma/.")

    os.system(f"rsync -r ./source/simulator/ns3_model/ipv4-interface.* {params['ns3_sim_path'].rstrip('/')}/src/internet/model/.")

    # raise Exception("STOP")
    ## go to ns3 dir
    os.chdir(params["ns3_sim_path"])

    ## run ns3 configure
    # configure_command = './waf -d optimized configure'
    if configure:
        os.system("./waf configure")
    ## run NS3 simulator
    ns3_params_format = (
        "prisma --simSeed={} --openGymPort={} --simTime={} --AvgPacketSize={} "
        "--LinkDelay={} --LinkRate={} --MaxBufferLength={} --load_factor={} "
        "--adj_mat_file_name={} --overlay_mat_file_name={} --node_coordinates_file_name={} "
        "--node_intensity_file_name={} --signaling={} --AgentType={} --signalingType={} "
        "--syncStep={} --lossPenalty={} "
        "--train={} --movingAverageObsSize={} --activateUnderlayTraffic={} "
        "--map_overlay_file_name={} --pingAsObs={} --NNSize={} --pingPacketIntervalTime={} --DT_SendAllDestinations={} --tunnels_max_delays_file_name={} --controller_id={} --perturbations={} --monitoring_type={}".format(
            params["seed"],
            params["basePort"],
            str(params["episode_duration"]),
            params["packet_size"],
            str(params["link_delay"]) + "ms",
            str(params["link_cap"]) + "bps",
            str(params["max_out_buffer_size"]) + "B",
            params["load_factor"],
            params["physical_adjacency_matrix_path"],
            params["overlay_adjacency_matrix_path"],
            params["node_coordinates_path"],
            params["traffic_matrix_path"],
            int(params["signalingSim"]),
            params["agent_type"],
            params["signaling_type"],
            params["target_update_period"],
            params["fixed_loss_penalty"],
            int(params["train"]),
            params["movingAverageObsSize"],
            int(params["activateUnderlayTraffic"]),
            params["map_overlay_path"],
            int(params["pingAsObs"]),
            params["nn_size"],
            params["pingPacketIntervalTime"],
            params["d_t_send_all_destinations"],
            params["tunnels_max_delays_file_name"],
            params["controller_id"],
            params["perturbations"],
            params["monitoring_type"]
        )
    )
    print("Running ns3 simulator with the following parameters:")
    print(ns3_params_format)
    run_ns3_command = shlex.split(f'./waf --run "{ns3_params_format}"')
    proc = subprocess.Popen(run_ns3_command)
    print(f"Running ns3 simulator with process id: {proc.pid}")
    os.chdir(current_folder_path)
    return proc.pid
