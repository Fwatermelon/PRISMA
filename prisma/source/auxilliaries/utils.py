# trunk-ignore-all(black)
# -*- coding: utf-8 -*-
import math
import os
import random
import subprocess
import numpy as np
import pandas as pd
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf


__author__ = (
    "Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
)
__copyright__ = "Copyright (c) 2022 Redha A. Alliche, Tiago Da Silva Barros, Ramon Aparicio-Pardo, Lucile Sassatelli"
__license__ = "GPL"
__email__ = "alliche,raparicio,sassatelli@i3s.unice.fr, tiago.da-silva-barros@inria.fr"



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


def convert_data_rate_to_bps(data_rate):
    """
    Convert the int data rate into text

    Args:
        data_rate (int): data rate in bps
    """
    if data_rate == 0:
        return "0bps"
    size_name = ("bps", "Kbps", "Mbps", "Gbps", "Tbps")
    i = int(math.floor(math.log(data_rate, 1000)))
    p = math.pow(1000, i)
    s = round(data_rate / p, 2)
    return "%s%s" % (s, size_name[i])


def convert_bps_to_data_rate(bps):
    """
    Convert the text data rate into int

    Args:
        bps (str): data rate in text
    """
    unit = bps.rstrip("bps")
    size_name = ("K", "M", "G", "T")
    if unit[-1] in size_name:
        i = size_name.index(unit[-1])
        p = math.pow(1000, i + 1)
        data_rate = float(unit[:-1])
    else:
        p = 1
        data_rate = float(unit)

    return data_rate * p

def allocate_on_gpu(gpu_memory_margin=1500)->bool:
    """
    Determine which gpu to use based on the available memory (works only for nvidia gpus)
    Args:
        gpu_memory_margin (int, optional): The margin of the gpu memory. Defaults to 1500.
    Returns:
        bool: True if a gpu is available, False otherwise
    """
    # check the available gpu device //from https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python


    # check if gpu is available
    if len(tf.config.list_physical_devices("GPU")) == 0:
        print("No gpu available")
        gpu_index = -1
        return
    try:
        # set the margin of the gpu memory
        gpu_memory_margin = 1500  # required memory but a train instance in MB
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        output = subprocess.getoutput(command).split("\n")[1:]
        # output = sp.check_output(command).decode("utf-8").split("\n")[1:-1]
    except Exception as e:
        print("Not able to get the gpu usage", e)
        return False
    # gpu_usage = [int(x.split(" ")[0]) for x in output]
    available_memory = [int(x.split(" ")[0]) for x in output]
    # get the gpu with the most available memory
    if max(available_memory) < gpu_memory_margin:
        print("No gpu available")
        gpu_index = -1
    else:
        gpu_index = np.argmax(available_memory)
        print("Available gpu memory : ", max(available_memory), " MB", "GPU index : ", gpu_index)
    # allocate the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    print("GPU index : ", gpu_index)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_index], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        except RuntimeError as e:
            print(e)

    # tf.config.set_soft_device_placement(True)
    return True


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    sources :
        - https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
        - https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file

    """

    def convert_tfevent(filepath):
        return pd.DataFrame(
            [
                parse_tfevent(e)
                for e in summary_iterator(filepath)
                if len(e.summary.value)
            ]
        )

    def parse_tfevent(tfevent):
        if "hparams" in tfevent.summary.value[0].tag:
            scalar = 0.0
        else:
            scalar = tensor_util.MakeNdarray(tfevent.summary.value[0].tensor).item()
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=scalar,
        )

    columns_order = ["wall_time", "name", "step", "value"]

    out = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


def fix_seed(seed):
    """Fix the seed for all random generators.

    Parameters:
        seed (int): the seed to use.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
