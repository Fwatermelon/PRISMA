"""
This script plots the results of a test session.
It will go through all the test files in a given directory and plot: 
    - The average cost for each sync step (cost vs. sync step)
    - The training overhead for each sync step (training overhead vs. sync step)
    - The average cost and overhead for each sync step (cost vs. overhead)
    - The average cost for each load (cost vs. load)
    - The end-to-end delay for each load (delay vs. load)
    - The loss rate for each load (loss rate vs. load)
It will retrive the information from the tensorboard files.
It will compare the models with shortest path routing and optimal routing as benchmark.

author: allicheredha@gmail.com
"""

# ------------------------------------------------------------
# %%
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("/home/redha/prisma_chap4/prisma/source/auxilliaries")
from utils import convert_tb_data


def plot_metric_vs_sync_step(result_df, exps_name, metric_name, axis):
    # plot the metric vs. sync step given a dataframes of results
    data = result_df[result_df.name == metric_name].sort_values("step")
    x = data.step.values
    y = data.value.values
    axis.plot(x, y, label=exps_name)


# The directory where the test results are stored
dir_path = "/home/redha/prisma_chap4/prisma/examples/5n_overlay_full_mesh_abilene/results/Final_results"
# dir_path = '/mnt/journal_paper_results/ex/5n_overlay_full_mesh_abilene/results/Final_results'
dir_path = "/mnt/backup_examples_new/5n_overlay_full_mesh_abilene/results/abilene_results_with_threshold"
dir_path = "/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold"
# dir_path = '/mnt/final_results_geant'
dir_path = "/home/redha/prisma_chap4/prisma/examples/abilene/results/abilene_results"
plot_style = {
    "font.family": "Times New Roman",
    "font.size": 50,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 3,
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.facecolor": "#EBEBEB",
    "xtick.color": "black",
    "ytick.color": "black",
    "ytick.minor.visible": False,
    "xtick.minor.visible": False,
    "grid.color": "black",
    "grid.linestyle": "-.",
    "grid.linewidth": 0.6,
    "figure.autolayout": True,
    "figure.figsize": (19.4, 10),
}

plt.rcParams.update(plot_style)
# evaluation metric name
metric_names = {
    "stats": [
        "avg_hops_over_time",
        "signalling ratio",
        "total_hops_over_time",
        "all_signalling_ratios",
    ],
    "nb_new_pkts": ["pkts_over_time"],
    "nb_lost_pkts": ["pkts_over_time"],
    "nb_arrived_pkts": ["pkts_over_time"],
    "test_results": [
        "test_overlay_loss_rate",
        "test_overlay_e2e_delay",
        #  "test_global_loss_rate",
        #  "test_global_e2e_delay"
    ],
}
metric_name = "test_overlay_loss_rate"
exps_names = []

# ------------------------------------------------------------
# % set the parameters
# static parameters
traff_mats = [0, 2, 3, 1]
# traff_mats = list(range(11))
traff_mats = [0,]
sync_steps = list(range(1, 8))
sync_steps = list(range(1, 10)) + list(range(10, 22, 2))
sync_steps = list(range(1, 10, 2))
# sync_steps.remove(1)
# sync_steps.remove(3)
sync_steps = [1,]
seed = 100
rb_size = 10000
signaling_types = ["dqn_logit_sharing"]
# signaling_types = ["dqn_model_sharing", "sp", "opt_10"]
# signaling_types = ["dqn_model_sharing",]
# dqn_models = ["", "_lite", "_lighter", "_lighter_2", "_lighter_3", "_ff"][:-1]
dqn_models = [
    "",
]
nn_size = 35328
# variable parameters
training_loads = [0.4, 0.7, 0.9]
training_loads = [0.4]
topology_name = "abilene"
bs = 512
lr = 0.0001
exploration = ["vary", 1.0, 0.01]
ping_freq = 10000
mv_avg_interval = 5
max_output_buffer_size = 16260
train_duration = 20
test_loads = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
loss_penalty_types = ["fixed", "None"]
name_to_save = [ "None"]
lambda_train_step = -1
buffer_soft_limit = 0
snapshots = ["final"]
# snapshots = ["final"] + [f"episode_{i}_step_{i}" for i in range(1, 20, 1)]
ping_freqs = [0.1, 0.5, 1]
ping_freqs = [0.1,]
# snapshots =  ["final"]

# % reset the variables
all_exps_results = {}
# go through all the experiments
found = []
thresholds = [0.0, 0.25, 0.5, 0.75]
thresholds = [0.0]
loss_flag = 1
use_loss_flag = 1
optimal_root_path = (
    "/mnt/backup_examples_new/5n_overlay_full_mesh_abilene/optimal_solution/"
)


def load_data(path, metric_name, case):
    exp_df = convert_tb_data(path)
    if case in ["stats", "nb_new_pkts", "nb_lost_pkts", "nb_arrived_pkts"]:
        if metric_name in ("all_signalling_ratios",):
            # compute average signalling ratio per episode
            exp_df.step = pd.to_datetime(exp_df.step / 1000, unit="ms")
            exp_df = exp_df.resample("20s", on="step", closed="right").mean()
            exp_df.step = exp_df.index
        else:
            exp_df = exp_df.iloc[-1]
    steps, indices = np.unique(exp_df.step, return_index=True)
    if len(steps) > 1:
        np.array(exp_df.value)[indices].reshape(-1, 1)
    else:
        np.array(exp_df.value).reshape(-1, 1)


# ------------------------------------------------------------
# %% retrieve the results
not_found = []
# all_exps_results = np.load(
#     "/home/redha/PRISMA_copy/prisma/all_exps_results_old_geant.npy", allow_pickle=True
# ).item()
# loss_penalty_types = ["constrained" ]
# signaling_types = ["dqn_model_sharing",]
# training_loads = [0.9]
# snapshots = ["final"]
for signaling_type in signaling_types:
    for idx, dqn_model in enumerate(dqn_models):
        for iii, loss_penalty_type in enumerate(loss_penalty_types):
            for training_load in training_loads:
                for ping_freq in ping_freqs:
                    for threshold in thresholds:
                        for snapshot in snapshots:
                            if signaling_type != "dqn_model_sharing" and dqn_model != "":
                                continue
                            for traff_mat in traff_mats:
                                for sync_step in sync_steps:
                                    if signaling_type != "dqn_model_sharing":
                                        if (
                                            sync_step != 1
                                        ):
                                            continue
                                    if ping_freq != 0.1:
                                        if (
                                            training_load != 0.9
                                            or sync_step != 1
                                            or signaling_type != "dqn_model_sharing"
                                            or threshold != 0.0
                                            or snapshot != "final"
                                        ):
                                            continue
                                    if snapshot != "final":
                                        if not (
                                            sync_step in (1,) and training_load == 0.9
                                        ):
                                            continue
                                    if training_load != 0.9 and sync_step > 9:
                                        continue
                                    session_name = f"sync_{sync_step}_mat_{traff_mat}_dqn__{signaling_type}_size_{nn_size}_tr_{training_load}_sim_60_1_lr_0.0001_bs_512_outb_16260_losspen_{loss_penalty_type}_vary_reset_{0}_threshold_{threshold}"

                                    if signaling_type in [
                                        "sp",
                                        "opt",
                                        "opt_9",
                                        "opt_10",
                                        "opt_11",
                                        "opt_12",
                                        "opt_13",
                                        "opt_14",
                                        "opt_15",
                                    ]:
                                        if signaling_type == "sp":
                                            session_name = f"{signaling_type}_{traff_mat}_25_final_ping_1000"
                                        else:
                                            session_name = f"opt_{traff_mat}_25_new_ping_1000_{signaling_type[4:]}"

                                        snapshot = ""
                                    if f"{session_name}_{snapshot}" in found:
                                        continue
                                    exps_names.append(f"{session_name}_{snapshot}")
                                    for case in metric_names.keys():
                                        if case not in [
                                            "test_results",
                                        ] and signaling_type in (
                                            "sp",
                                            "opt",
                                            "opt_9",
                                            "opt_10",
                                            "opt_11",
                                            "opt_12",
                                            "opt_13",
                                            "opt_14",
                                            "opt_15",
                                        ):
                                            continue
                                        if case not in [
                                            "test_results",
                                        ]:
                                            path = f"{dir_path}/{session_name}/{case}"
                                        else:
                                            path = f"{dir_path}/{session_name}/{case}/{snapshot}"
                                        try:
                                            if not os.path.exists(path):
                                                raise (
                                                    Exception(
                                                        f"{case} results not found"
                                                    )
                                                )
                                            # Get the results for the experiment
                                            exp_df = convert_tb_data(path)
                                            for metric_name in metric_names[case]:
                                                if (
                                                    metric_name
                                                    == "all_signalling_ratios"
                                                ):
                                                    data = exp_df[
                                                        exp_df.name
                                                        == "signalling ratio"
                                                    ].sort_values("step")
                                                else:
                                                    data = exp_df[
                                                        exp_df.name == metric_name
                                                    ].sort_values("step")

                                                if case not in [
                                                    "test_results",
                                                ]:
                                                    if metric_name in (
                                                        "all_signalling_ratios",
                                                    ):
                                                        # compute average signalling ratio per episode
                                                        data.step = pd.to_datetime(
                                                            data.step / 1000, unit="ms"
                                                        )
                                                        data = data.resample(
                                                            "20s",
                                                            on="step",
                                                            closed="right",
                                                        ).mean()
                                                        data.step = data.index
                                                    else:
                                                        data = data.iloc[-1]
                                                steps, indices = np.unique(
                                                    data.step, return_index=True
                                                )
                                                if len(steps) > 1:
                                                    data_to_store = np.array(
                                                        data.value
                                                    )[indices].reshape(-1, 1)
                                                else:
                                                    data_to_store = np.array(
                                                        data.value
                                                    ).reshape(-1, 1)
                                                key = f"{signaling_type}{dqn_model}_{sync_step}_{name_to_save[iii]}_{training_load}_{ping_freq}_{snapshot}_{case}_{metric_name}_{threshold}"
                                                if signaling_type in [
                                                    "sp",
                                                    "opt",
                                                    "opt_9",
                                                    "opt_10",
                                                    "opt_11",
                                                    "opt_12",
                                                    "opt_13",
                                                    "opt_14",
                                                    "opt_15",
                                                ]:
                                                    key = f"{signaling_type}_{traff_mat}_{case}_{metric_name}"
                                                if key not in all_exps_results.keys():
                                                    all_exps_results[key] = {
                                                        "loads": steps.tolist(),
                                                        "traff_mat": [
                                                            traff_mat,
                                                        ],
                                                        "data": data_to_store,
                                                    }
                                                else:
                                                    if (
                                                        traff_mat
                                                        not in all_exps_results[key][
                                                            "traff_mat"
                                                        ]
                                                    ):
                                                        all_exps_results[key][
                                                            "data"
                                                        ] = np.concatenate(
                                                            (
                                                                all_exps_results[key][
                                                                    "data"
                                                                ],
                                                                data_to_store,
                                                            ),
                                                            axis=1,
                                                        )
                                                        all_exps_results[key][
                                                            "traff_mat"
                                                        ].append(traff_mat)
                                                if case == "test_results":
                                                    assert len(
                                                        exp_df.step.unique()
                                                    ) >= len(
                                                        test_loads
                                                    )  # check if all the test loads are present
                                        except Exception:
                                            print(session_name, snapshot, path)
                                            not_found.append(
                                                f"{session_name}_{snapshot}"
                                            )
                                            continue
                                        found.append(f"{session_name}_{snapshot}")


# ------------------------------------------------------------
# %% recover opt results
# all_exps_results = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results.npy", allow_pickle=True).item()
# all_exps_results_old = np.load("/home/redha/PRISMA_copy/prisma/all_exps_results_old.npy", allow_pickle=True).item()

for name in [
    "opt",
]:
    for traff_mat in traff_mats:
        data_loss = []
        data_delay = []
        # pth = f"/mnt/backup_examples_new/5n_overlay_full_mesh_abilene/results/abilene_results_with_threshold/opt_{traff_mat}_25_final_ping_1000"
        # pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{name}_{traff_mat}_25_final_ping_1000_threshold_0.0"
        pth = f"/mnt/backup_examples_new/overlay_full_mesh_10n_geant/results/geant_results_with_threshold/{name}_{traff_mat}_25_final_ping_1000"
        # for load in test_loads:
        # temp_dict = json.load(open(f"{optimal_root_path}/{traff_mat}_norm_matrix_uniform/{int(load*100)}_ut_minCostMCF.json"))
        # data_delay.append(temp_dict["avg_hops"]*((540*8/500000)+0.001))
        # data_loss.append(np.round(temp_dict["lost_traff"], 4)/temp_dict["offered_traff"])
        exp_df = convert_tb_data(pth)
        for metric_name in metric_names["test_results"]:
            data = exp_df[exp_df.name == metric_name].sort_values("step")
            steps, indices = np.unique(data.step, return_index=True)
            if len(steps) > 1:
                data_to_store = np.array(data.value)[indices].reshape(-1, 1)
            else:
                data_to_store = np.array(data.value).reshape(-1, 1)
            key = f"{name}___{traff_mat}_test_results_{metric_name}"
            if key not in all_exps_results.keys():
                all_exps_results[key] = {
                    "loads": steps.tolist(),
                    "traff_mat": [
                        traff_mat,
                    ],
                    "data": data_to_store,
                }
            else:
                if traff_mat not in all_exps_results[key]["traff_mat"]:
                    all_exps_results[key]["data"] = np.concatenate(
                        (all_exps_results[key]["data"], data_to_store), axis=1
                    )
                    all_exps_results[key]["traff_mat"].append(traff_mat)

        # key = f"opt_{traff_mat}_test_results_test_overlay_e2e_delay"
        # all_exps_results[key] = {"loads": test_loads,
        #                         "traff_mat": [traff_mat,],
        #                         "data": np.array(data_delay).reshape(1, -1)}
        # key = f"opt_{traff_mat}_test_results_test_overlay_loss_rate"
        # all_exps_results[key] = {"loads": test_loads,
        #                         "traff_mat": [traff_mat,],
        #                         "data": np.array(data_loss).reshape(1, -1)}

# 1)------------------------------------------------------------
# %% box plot for the metrics vs train load over all sync steps and traffic matrices for each model
ping_freq = 0.1
threshold = 0.0

for metric in metric_names["test_results"]:
    fig, ax = plt.subplots()
    # add dqn buffer dqn_model_sharing model
    for snapshot in ["final"]:
        for tr_idx, training_load in enumerate(training_loads):
            bps = []
            for model_idx, loss_penalty_type in enumerate(
                ["None", "fixed", "constrained"]
            ):
                y = []
                y_min = []
                y_max = []
                x = []
                for sync_step in sync_steps:
                    name = f"dqn_model_sharing{''}_{sync_step}_{loss_penalty_type}_{training_load}_{ping_freq}_{snapshot}_test_results"
                    temp = np.mean(
                        all_exps_results_old[f"{name}_{metric}_{threshold}"]["data"][
                            :, :
                        ],
                        axis=0,
                    )
                    y.append(np.mean(temp))
                    # y_min.append(np.mean(temp) - np.std(temp))
                    # y_max.append(np.mean(temp) + np.std(temp))
                    # x.append(sync_step)
                # add box plot for each model for each training load, put the training load on the x axis and the metric on the y axis and put each model in a different color
                # plt.plot(y, label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"), marker="o")
                bps.append(
                    plt.boxplot(
                        y,
                        positions=[tr_idx + model_idx * 0.3],
                        widths=0.25,
                        showfliers=False,
                        patch_artist=True,
                        boxprops=dict(facecolor=f"C{model_idx}"),
                        medianprops=dict(color="black"),
                    )
                )

    # add sp and opt
    y = []
    for mat_idx in traff_mats:
        y.append(
            np.mean(
                all_exps_results_old[f"sp_{mat_idx}_test_results_{metric}"]["data"],
                axis=0,
            )[0]
        )
    sp_plot = plt.hlines(
        np.mean(y), -0.2, 2.8, label="OSPF", linestyle="--", color="red", linewidth=7
    )
    # plt.fill_between([-0.2, 2.8], np.min(y, axis=0), np.max(y, axis=0), alpha=0.1, color="red")
    y = []
    for mat_idx in traff_mats:
        y.append(
            np.mean(
                all_exps_results_old[f"opt_{mat_idx}_test_results_{metric}"]["data"],
                axis=0,
            )[0]
        )
    opt_plot = plt.hlines(
        np.mean(y), -0.2, 2.8, label="OPT", linestyle="--", color="green", linewidth=7
    )
    metric_name = (
        f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost")
        .replace("lossrate", "Packet Loss Ratio")
        .replace("delay", " Delay (ms)")
        .replace("overlay", "")
        .replace("global", "Global ")
        .replace("e2e", "End-to-End ")
    )
    plt.xlabel(xlabel="Traffic Load during Training")
    plt.ylabel(ylabel=f"{metric_name}")
    plt.xticks(ticks=[0.3, 1.3, 2.3], labels=["40%", "70%", "90%"])
    if metric == "test_overlay_loss_rate":
        plt.yticks(
            ticks=[0.0, 0.01, 0.02, 0.03, 0.04], labels=["0%", "1%", "2%", "3%", "4%"]
        )
    else:
        plt.yticks(ax.get_yticks(), np.array(ax.get_yticks() * 1000, dtype=int))
    # plt.legend(loc="upper left", ncols=6, handles=[bp["boxes"][0] for bp in bps] + [sp_plot, opt_plot])
    # plt.title(f"The effect of training load in Offband setting")
    import tikzplotlib

    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{metric_name}_vs_training_loads.tex", axis_width="\\figwidth", axis_height="\\figheight", extra_axis_parameters={"mark options={scale=2}"})
    # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{metric_name}_vs_training_loads.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)
# generate empty figure to include only the legend
plt.figure()
plt.axis("off")
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# add legend
plt.legend(
    [bp["boxes"][0] for bp in bps] + [sp_plot, opt_plot],
    [
        "Loss Blind",
        "Fixed Loss-Pen",
        "Guided Reward",
        "Shortest Path",
        "Oracle Routing",
    ],
    loc="upper left",
    ncols=6,
)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
# plt.tight_layout()
plt.savefig(
    f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_legend.pdf".replace(
        " ", "_"
    ),
    bbox_inches="tight",
    pad_inches=0.1,
)

# 2)------------------------------------------------------------
# %% Perf 2: plot the results & save the figures for cost, loss and delay vs sync step
figures_dir = "/home/redha/PRISMA_copy/prisma/figures"
costs = {}

max_sync_step = 10
for metric in metric_names["test_results"]:
    fig, ax = plt.subplots()
    plots = []
    # add dqn buffer dqn_model_sharing model
    for training_load in [0.9]:
        for threshold in [0.0]:
            for snapshot in ["final"]:
                for loss_penalty_type in ["None", "fixed", "constrained"]:
                    for model in dqn_models:
                        y = []
                        y_min = []
                        y_max = []
                        x = []
                        for sync_step in sync_steps[:max_sync_step]:
                            name = f"dqn_model_sharing{model}_{sync_step}_{loss_penalty_type}_{training_load}_{0.1}_{snapshot}_test_results"
                            temp = np.mean(
                                all_exps_results_old[f"{name}_{metric}_{threshold}"][
                                    "data"
                                ][:, :],
                                axis=0,
                            )
                            y.append(np.mean(temp))
                            y_min.append(np.mean(temp) - np.std(temp))
                            y_max.append(np.mean(temp) + np.std(temp))
                            x.append(sync_step)
                        # add the plot into the figure with min and max values shaded area
                        if loss_penalty_type == "constrained":
                            costs[f"{metric}_NN_{threshold}"] = y
                        plots.append(
                            plt.plot(
                                x,
                                y,
                                label=f"{loss_penalty_type}".replace(
                                    "None", "Loss Blind"
                                )
                                .replace("fixed", "Loss Aware")
                                .replace("constrained", "RCPO")
                                .replace("digital twin", "Digital Twin"),
                                linestyle="solid",
                                marker="o",
                                linewidth=7,
                                markersize=20,
                            )[0]
                        )
                        ax.fill_between(x, y_min, y_max, alpha=0.2)
                        # ax.plot(x, y_min,
                        #         label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"),
                        #             linestyle="dashed",
                        #          marker="o",
                        #          linewidth=7,
                        #          markersize=20)
                        # ax.plot(x, y_max,
                        #         label=f"{loss_penalty_type}_{training_load}".replace("None", "Loss Blind").replace("fixed", "Loss Aware").replace("constrained", "RCPO"),
                        #         linestyle="dashed",
                        #          marker="o",
                        #          linewidth=7,
                        #          markersize=20)
                    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
    thresholds = np.sort(thresholds)[::-1]
    for threshold in thresholds:
        # add digital twin
        temp = np.mean(
            all_exps_results[
                f"digital_twin_1_constrained_0.9_0.1_final_test_results_test_overlay_e2e_delay_{threshold}"
            ]["data"],
            axis=0,
        )
        costs[f"test_overlay_e2e_delay_digital_twin_{threshold}"] = np.mean(temp)
        temp = np.mean(
            all_exps_results[
                f"digital_twin_1_constrained_0.9_0.1_final_test_results_test_overlay_loss_rate_{threshold}"
            ]["data"],
            axis=0,
        )
        # ax.hlines(np.mean(temp), 1, 20, label=f"digital twin_{threshold}", linestyle="--", linewidth=7)
        costs[f"test_overlay_loss_rate_digital_twin_{threshold}"] = np.mean(temp)
        # # # add digital twin
        # # temp = np.mean(all_exps_results[f"digital_twin_1_5_{metric}"]["data"], axis=0)
        # # ax.hlines(np.mean(temp), 1, 8, label="DT 5", linestyle="--")
        # # add dqn_value_sharing
        temp = np.mean(
            all_exps_results[
                f"target_1_constrained_0.9_0.1_final_test_results_test_overlay_e2e_delay_{threshold}"
            ]["data"],
            axis=0,
        )
        costs[f"test_overlay_e2e_delay_target_{threshold}"] = np.mean(temp)
        temp = np.mean(
            all_exps_results[
                f"target_1_constrained_0.9_0.1_final_test_results_test_overlay_loss_rate_{threshold}"
            ]["data"],
            axis=0,
        )
        costs[f"test_overlay_loss_rate_target_{threshold}"] = np.mean(temp)
        # ax.hlines(np.mean(temp), 1, 20, label=f"target_{threshold}", linestyle="--", linewidth=7)

    # add sp and opt
    y = []
    for tr_idx in traff_mats:
        y.append(
            np.mean(
                all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"], axis=0
            )[0]
        )
    sp_plot = plt.hlines(
        np.mean(y),
        min(sync_steps),
        max(sync_steps[:max_sync_step]),
        label="OSPF",
        linestyle="--",
        color="red",
        linewidth=7,
    )
    # ax.fill_between(sync_steps, np.min(y, axis=0), np.max(y, axis=0), alpha=0.2, color="red")

    y = []
    for tr_idx in traff_mats:
        y.append(
            np.mean(
                all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"], axis=0
            )
        )
    opt_plot = plt.hlines(
        np.mean(y),
        min(sync_steps),
        max(sync_steps[:max_sync_step]),
        label="Oracle Routing",
        linestyle="--",
        color="tab:purple",
        linewidth=7,
    )

    plt.xlabel(xlabel="dqn_value_sharing Update Period U (s)")
    metric_name = (
        f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost")
        .replace("lossrate", "Packet Loss Ratio")
        .replace("delay", " Delay (ms)")
        .replace("overlay", "")
        .replace("global", "Global ")
        .replace("e2e", "End-to-End ")
    )
    plt.ylabel(ylabel=f"{metric_name}")
    if metric == "test_overlay_loss_rate":
        plt.yticks(
            ticks=[0.0, 0.01, 0.02, 0.03, 0.04], labels=["0%", "1%", "2%", "3%", "4%"]
        )
        plt.ylim(0, 0.03)
    else:
        plt.yticks(ax.get_yticks(), np.array(ax.get_yticks() * 1000, dtype=int))
    # plt.legend(loc="upper left")
    plt.xticks(sync_steps[:max_sync_step])
    # plt.title(f"The effect of sync step in Offband setting for 90% training load")

    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_name}_vs_sync_steps.tex", axis_width="\\figwidth", axis_height="\\figheight", extra_axis_parameters={"mark options={scale=2}"})

    # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_name}_vs_sync_steps.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)

# add legend 2
# generate empty figure to include only the legend
plt.figure(figsize=(19.4, 10))
plt.axis("off")
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.legend(
    plots + [sp_plot, opt_plot],
    [
        "Loss Blind",
        "Fixed Loss-Pen",
        "Guided Reward",
        "Shortest Path",
        "Oracle Routing",
    ],
    ncols=6,
)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
# plt.tight_layout()
plt.savefig(
    f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_legend_2.pdf".replace(
        " ", "_"
    ),
    bbox_inches="tight",
    pad_inches=0.1,
)

# 3) ------------------------------------------------------------
# %% Perf 3:  show the effect of ping freq for the best config in a table
for metric in metric_names["test_results"]:
    print(metric)
    for threshold in [0.0]:
        print(f"threshold: {threshold}")
        for training_load in [0.9]:
            for snapshot in ["final"]:
                for loss_penalty_type in ["constrained"]:
                    for sync_step in [1]:
                        for ping_freq in ping_freqs[:]:
                            name = f"dqn_model_sharing{''}_{sync_step}_{loss_penalty_type}_{training_load}_{ping_freq}_{snapshot}_test_results"
                            temp = np.mean(
                                all_exps_results_old[f"{name}_{metric}_{threshold}"][
                                    "data"
                                ][:, :]
                            )
                            print(f"ping_freq: {ping_freq}, {temp*100}")

# 4)------------------------------------------------------------
# %% Convergence 1 : plot the metrics vs episode for each loss_penalty_type, for dqn_model_sharing model sync step 1 and train load 0.9
metric_names_str = ["loss", "delay"]
for metric_index, metric in enumerate(metric_names["test_results"]):
    fig, ax = plt.subplots()
    for threshold in [
        0.0,
    ]:
        for loss_penalty_type in loss_penalty_types[:]:
            for signaling_type in [
                "dqn_model_sharing",
            ]:
                # add dqn buffer dqn_model_sharing model
                y = []
                y_min = []
                y_max = []
                x = []
                for i, snapshot in enumerate(
                    [f"episode_{i}_step_{i}" for i in range(1, 20, 1)] + ["final"]
                ):
                    name = f"{signaling_type}{''}_{1}_{loss_penalty_type}_{0.9}_{0.1}_{snapshot}_test_results"
                    temp = np.mean(
                        all_exps_results_old[f"{name}_{metric}_{threshold}"]["data"],
                        axis=0,
                    )
                    y.append(np.mean(temp))
                    y_min.append(np.mean(temp) - np.std(temp))
                    y_max.append(np.mean(temp) + np.std(temp))
                    x.append(i)
                # add the plot into the figure with min and max values shaded area
                ax.plot(
                    x,
                    y,
                    label=f"{signaling_type} {loss_penalty_type} {threshold}".replace(
                        "None", "Loss Blind"
                    )
                    .replace("fixed", "Loss Aware")
                    .replace("constrained", "RCPO")
                    .replace("dqn_model_sharing", "Model Sharing offline")
                    .replace("dqn_logit_sharing", "DT"),
                    marker="o",
                    linewidth=7,
                    markersize=20,
                )
                # ax.fill_between(x, y_min, y_max, alpha=0.2)

        plt.xlabel(xlabel="Episode number")
        metric_name = (
            f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost")
            .replace("lossrate", "Packet Loss Ratio")
            .replace("delay", " Delay (ms)")
            .replace("overlay", "")
            .replace("global", "Global ")
            .replace("e2e", "End-to-End ")
        )
        plt.ylabel(ylabel=f"{metric_name}")
        if metric == "test_overlay_loss_rate":
            plt.yticks(
                ticks=[0.0, 0.01, 0.02, 0.03, 0.04],
                labels=["0%", "1%", "2%", "3%", "4%"],
            )
        else:
            plt.yticks(ax.get_yticks(), np.array(ax.get_yticks() * 1000, dtype=int))
        # plt.legend(loc="upper left")

        # add sp and opt
        y = []
        for tr_idx in traff_mats:
            y.append(
                np.mean(
                    all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"],
                    axis=0,
                )[0]
            )
        ax.hlines(
            np.mean(y), 0, 20, label="OSPF", linestyle="--", color="red", linewidth=7
        )
        # ax.fill_between(sync_steps, np.min(y, axis=0), np.max(y, axis=0), alpha=0.2, color="red")
        y = []
        for tr_idx in traff_mats:
            y.append(
                np.mean(
                    all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"],
                    axis=0,
                )
            )
        ax.hlines(
            np.mean(y),
            0,
            20,
            label="Oracle Routing",
            linestyle="--",
            color="tab:purple",
            linewidth=7,
        )
        # plt.xticks(ax.get_xticks(), ax.get_xticks()+1)
        # plt.tight_layout()
    tikzplotlib.clean_figure()
    tikzplotlib.save(
        f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_names_str[metric_index]}_vs_sync_steps.tex",
        axis_width="\\figwidth",
        axis_height="\\figheight",
        extra_axis_parameters={"mark options={scale=2}"},
    )

    plt.savefig(
        f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_names_str[metric_index]}_vs_sync_steps.pdf".replace(
            " ", "_"
        ),
        bbox_inches="tight",
        pad_inches=0.1,
    )

# 5)------------------------------------------------------------
# %% Overhead 1 : plot the results & save the figures for overhead vs sync step
overheads = {
    "dqn_model_sharing": [],
    "dqn_logit_sharing": [],
    "dqn_value_sharing": [],
}
fig, ax = plt.subplots()
# add dqn buffer dqn_model_sharing model
for signaling_type in signaling_types[:-2]:
    for loss_penalty_type in ["constrained"]:
        for threshold in thresholds:
            for training_load in [0.9]:
                y = []
                y_min = []
                y_max = []
                x = []
                for sync_step in sync_steps[:max_sync_step]:
                    if signaling_type != "dqn_model_sharing":
                        temp = all_exps_results[
                            f"{signaling_type}_{1}_{loss_penalty_type}_{training_load}_{0.1}_final_stats_signalling ratio_{threshold}"
                        ]["data"].max()
                    else:
                        temp = all_exps_results[
                            f"{signaling_type}_{1}_{loss_penalty_type}_{training_load}_{0.1}_final_stats_signalling ratio_{threshold}"
                        ]["data"].max() + (
                            (69 * 4 * 5 * 400 / sync_step)
                            / (
                                all_exps_results[
                                    f"{signaling_type}_{sync_step}_{loss_penalty_type}_{training_load}_{0.1}_final_nb_new_pkts_pkts_over_time_{threshold}"
                                ]["data"].max()
                                * 20
                            )
                        )
                    y.append(np.mean(temp))
                    y_min.append(np.mean(temp) - np.std(temp))
                    y_max.append(np.mean(temp) + np.std(temp))
                    x.append(sync_step)
                # add the plot into the figure with min and max values shaded area
                overheads[f"{signaling_type}_{threshold}"] = y
                ax.plot(
                    x,
                    y,
                    label=f"{signaling_type}_{threshold}".replace("dqn_model_sharing", "Model Sharing")
                    .replace("dqn_logit_sharing", "Digital Twin")
                    .replace("dqn_value_sharing", "dqn_value_sharing Value Sharing")
                    .replace("sp", "OSPF")
                    .replace("opt", "Oracle Routing"),
                    marker="o",
                    linewidth=7,
                    markersize=20,
                )
                ax.fill_between(x, y_min, y_max, alpha=0.2)
# add digital twin
# temp = (all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["dqn_model_sharing" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
# add dqn_value_sharing
# temp = (all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["dqn_model_sharing" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label=f"dqn_value_sharing", linestyle="--", color="grey")
plt.xlabel(xlabel="Sync Step")
plt.ylabel(ylabel="Overhead Ratio")
ax.set_xticks(sync_steps)
# list(np.round(np.unique(overheads['dqn_model_sharing']), 1)) + list(np.round(np.unique(overheads['dqn_logit_sharing']), 1)) + list(np.round(np.unique(overheads['dqn_value_sharing']), 1)))
# ax.set_yticks([0.3, 1,1.5, 2, 3, 4, 5.7])
plt.legend()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/overhead_ratio_vs_sync_steps.pdf",bbox_inches='tight', pad_inches=0.1)
# %%6)------------------------------------------------------------

# plot the results & save the figures for cost vs overhead ratio
thresholds = [0.0, 0.25, 0.5, 0.75]
min_x = 0.08
max_x = 1.4
max_data = 3
for metric_index, metric in enumerate(metric_names["test_results"]):
    fig, ax = plt.subplots(figsize=(18, 14))
    y = []
    x = []
    value_dt = []
    for threshold in thresholds:
        # add dqn buffer dqn_model_sharing model
        if threshold == 0.0:
            print(
                "dqn_model_sharing",
                metric,
                threshold,
                len(costs[f"{metric}_NN_{threshold}"]),
                len(overheads[f"NN_{threshold}"]),
            )
            nn_plot = ax.plot(
                overheads[f"NN_{threshold}"][1:],
                costs[f"{metric}_NN_{threshold}"][1:],
                label="Model-Sharing Offband",
                marker="o",
                linewidth=7,
                markersize=20,
                color="blue",
            )
        # add digital twin
        # if threshold == 0.0:
        #     dt_plot = ax.plot(overheads[f"digital_twin_{threshold}"][0], costs[f"{metric}_digital_twin_{threshold}"], label=f"Logit Sharing", marker="*", markersize=30, linewidth=0, color="green")
        # else:
        #     dt_plot = ax.plot(overheads[f"digital_twin_{threshold}"][0], costs[f"{metric}_digital_twin_{threshold}"], marker="*", markersize=30, linewidth=0, color="green")
        # print("dt", metric, threshold, np.mean(costs[f"{metric}_digital_twin_{threshold}"]))
        # if threshold == 0.0:
        value_dt.append(np.mean(costs[f"{metric}_digital_twin_{threshold}"]))
        y.append(costs[f"{metric}_digital_twin_{threshold}"])
        x.append(overheads[f"digital_twin_{threshold}"][0])
        # x.append(np.mean(all_exps_results[f"digital_twin_1_constrained_0.9_final_stats_all_signalling_ratios_{threshold}"]["data"], axis=1)[-1])
        # y.append(costs[f"{metric}_digital_twin_{threshold}"])
        ax.text(
            overheads[f"digital_twin_{threshold}"][0],
            costs[f"{metric}_digital_twin_{threshold}"],
            f"  ER_thr = {int(threshold*100)}%",
            fontsize=35,
            rotation=0,
            color="green",
            fontweight="bold",
        )
        # add dqn_value_sharing

        # ax.plot(overheads[f"target_{0.0}"][0], costs[f"{metric}_target_{0.0}"], label=f"dqn_value_sharing Value Sharing {0.25}", marker="*", markersize=30, linewidth=0,)
    our_model_plot = ax.plot(
        x, y, label="Our model", marker="*", markersize=30, color="green", linewidth=7
    )
    # add sp and opt
    sp_y = []
    for tr_idx in traff_mats:
        sp_y.append(
            np.mean(
                all_exps_results[f"sp_{tr_idx}_test_results_{metric}"]["data"], axis=0
            )[0]
        )
    sp_plot = ax.hlines(
        np.mean(sp_y),
        min_x,
        max_x,
        label="Shortest Path",
        linestyle="--",
        color="red",
        linewidth=7,
    )
    # ax.fill_between(sync_steps, np.min(y, axis=0), np.max(y, axis=0), alpha=0.2, color="red")
    opt_y = []
    for tr_idx in traff_mats:
        opt_y.append(
            np.mean(
                all_exps_results[f"opt_{tr_idx}_test_results_{metric}"]["data"], axis=0
            )[0]
        )
    for i, th in enumerate(thresholds):
        print("ratio sp", th, metric, (1 - (value_dt / np.mean(sp_y))))
        print(
            "ratio opt",
            th,
            metric,
            (value_dt[i] - np.mean(opt_y)) / (np.mean(opt_y) + value_dt[i]),
        )
    opt_plot = ax.hlines(
        np.mean(opt_y),
        0,
        6,
        label="Oracle Routing",
        linestyle="--",
        color="tab:purple",
        linewidth=7,
    )

    plt.xlabel(xlabel="Maximum Overhead Ratio during Training")
    metric_name = (
        f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost")
        .replace("lossrate", "Loss Rate")
        .replace("delay", " Delay")
        .replace("overlay", "Overlay ")
        .replace("global", "Global ")
    )
    plt.xlim(min_x, max_x)
    ticks = [0.15, 0.75, 1.0, 1.2] + x
    ax.set_xticks(ticks, [f"{int(x*100)}%" for x in ticks], rotation=90, fontsize=40)
    if metric == "test_overlay_loss_rate":
        plt.ylabel(ylabel="Loss Rate (%)")
        ax.set_yticks(
            list(np.arange(0, 0.030, 0.005)),
            [f"{np.round(x, 2)}%" for x in np.arange(0, 0.030, 0.005) * 100],
        )
    else:
        plt.ylabel(ylabel="Average End-to-End Delay (ms)")
        ax.set_yticks(
            list(np.arange(0.550, 0.590, 0.010)),
            [f"{int(x)}" for x in np.arange(0.550, 0.590, 0.010) * 1000],
        )
    # ax.legend( loc="upper right")
    # plt.tight_layout()
    # tikzplotlib.clean_figure()
    # tikzplotlib.save(
    #     f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_names_str[metric_index]}_vs_overhead.tex",
    #     axis_width="\\figwidth",
    #     axis_height="\\figheight",
    #     extra_axis_parameters={"mark options={scale=2}"},
    # )

    # plt.savefig(
    #     f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_{metric_names_str[metric_index]}_vs_overhead.pdf".replace(
    #         " ", "_"
    #     ),
    #     bbox_inches="tight",
    #     pad_inches=0.1,
    # )

# add legend 3
# generate empty figure to include only the legend
plt.figure(figsize=(19.4, 10))
plt.axis("off")
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.legend(
    nn_plot + our_model_plot + [sp_plot, opt_plot],
    ["Model Sharing off-band", "Our Framework", "Shortest Path", "Oracle Routing"],
    ncols=6,
)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
# plt.tight_layout()
# plt.savefig(
#     "/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/legend_3.pdf".replace(
#         " ", "_"
#     ),
#     bbox_inches="tight",
#     pad_inches=0.1,
# )

# 7)------------------------------------------------------------
# %% plot the overhead ratio vs episode for DT model sync step 1, train load 0.9, and rcpo loss penalty for each threshold
x = np.arange(1, 21, 1)
loss_penalty_type = "constrained"
metric = "all_signalling_ratios"
fig, ax = plt.subplots()
plots = []
for threshold in thresholds:
    # add dqn buffer dqn_model_sharing model
    name = f"dqn_logit_sharing{''}_{1}_{loss_penalty_type}_{0.9}_0.1_final_stats"
    y = np.mean(all_exps_results[f"{name}_{metric}_{threshold}"]["data"], axis=1)
    # y_min.append(np.mean(temp) - np.std(temp))
    # y_max.append(np.mean(temp) + np.std(temp))
    # x.append(i)
    # add the plot into the figure with min and max values shaded area
    plots.append(
        ax.plot(
            x,
            y,
            label=f"DT {threshold}".replace("None", "Loss Blind")
            .replace("fixed", "Loss Aware")
            .replace("constrained", "RCPO"),
            marker="o",
            linewidth=7,
            markersize=20,
        )[0]
    )
    # ax.fill_between(x, y_min, y_max, alpha=0.2)
plt.xlabel(xlabel="Episode Number")
metric_name = (
    f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost")
    .replace("lossrate", "Loss Rate")
    .replace("delay", " Delay")
    .replace("overlay", "Overlay ")
    .replace("global", "Global ")
)
# plt.ylim(0, 0.48)
plt.yticks([0.1, 0.2, 0.3, 0.4], [f"{int(x*100)}%" for x in [0.1, 0.2, 0.3, 0.4]])
plt.xticks(np.arange(1, 21, 6))
plt.ylabel(ylabel="Max Overhead Ratio\n during Training")
plt.tight_layout()
tikzplotlib.clean_figure()
tikzplotlib.save(
    "/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/overhead_vs_episodes.tex",
    axis_width="\\figwidth",
    axis_height="\\figheight",
    extra_axis_parameters={"mark options={scale=2}"},
)

plt.savefig(
    f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_overhead_vs_episodes.pdf".replace(
        " ", "_"
    ),
    bbox_inches="tight",
    pad_inches=0.1,
)

# add legend 4
# generate empty figure to include only the legend
plt.figure(figsize=(19.4, 10))
plt.axis("off")
# remove right and top margins and paddings
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.legend(plots, ["ER_thr=0%", "ER_thr=25%", "ER_thr=50%", "ER_thr=75%"], ncols=4)
# shrink the figure size to fit the box
plt.gcf().set_size_inches(0.1, 0.1, forward=True)
# remove the white space around the figure
# plt.tight_layout()
plt.savefig(
    f"/home/redha/PRISMA_copy/prisma/figures/journal_paper_1/{topology_name}_legend_4.pdf".replace(
        " ", "_"
    ),
    bbox_inches="tight",
    pad_inches=0.1,
)
# 3)------------------------------------------------------------
# %% plot the results & save the figures for cost, loss and delay vs load factors
figures_dir = "/home/redha/PRISMA_copy/prisma/figures"
threshold = 0.0
tr_mat = 6
for sync_step in [1]:
    # add dqn buffer dqn_model_sharing model
    for metric in metric_names["test_results"]:
        fig, ax = plt.subplots()
        for training_load in [0.9]:
            for snapshot in ["final"]:
                for signaling_type in [
                    "dqn_model_sharing",
                ]:
                    for loss_penalty_type in loss_penalty_types:
                        y = []
                        y_min = []
                        y_max = []
                        x = []
                        if loss_penalty_type == "constrained":
                            sync_step = 1
                        elif loss_penalty_type == "fixed":
                            sync_step = 3
                        elif loss_penalty_type == "None":
                            sync_step = 3
                        name = f"{signaling_type}{''}_{sync_step}_{loss_penalty_type}_{training_load}_{0.1}_{snapshot}_test_results"
                        # temp = np.mean(all_exps_results[f"{name}_{metric}"]["data"], axis=1)
                        temp = all_exps_results[f"{name}_{metric}_{threshold}"]["data"][
                            :, 1
                        ]
                        y.append(np.mean(temp))
                        y_min.append(np.mean(temp) - np.std(temp))
                        y_max.append(np.mean(temp) + np.std(temp))
                        x.append(sync_step)
                        # add the plot into the figure with min and max values shaded area
                        ax.plot(
                            all_exps_results[f"{name}_{metric}_{threshold}"]["loads"],
                            temp,
                            label=f"{loss_penalty_type}".replace("None", "Loss Blind")
                            .replace("fixed", "Loss Aware")
                            .replace("constrained", "RCPO"),
                            linestyle="dotted",
                            marker="o",
                            linewidth=7,
                            markersize=20,
                        )
                        # ax.fill_between(x, y_min, y_max, alpha=0.2)
        # add digital twin
        # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
        # add digital twin
        # temp = np.mean(all_exps_results[f"digital_twin_1_5_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
        # # add digital twin
        # temp = np.mean(all_exps_results[f"digital_twin_1_10_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label="DT 10", linestyle="--")
        # add dqn_value_sharing
        # temp = np.mean(all_exps_results[f"target_1_5_{metric}"]["data"], axis=0)
        # ax.hlines(np.mean(temp), 1, 8, label=f"dqn_value_sharing", linestyle="--", color="grey")

        # add sp and opt
        for name in ["sp", "opt"]:
            # for name in ["sp", "opt_9", ]:
            y = []
            for tr_idx in traff_mats:
                y.append(
                    all_exps_results[f"{name}_{tr_idx}_test_results_{metric}"]["data"][
                        :, 0
                    ]
                )
            ax.plot(
                all_exps_results[f"{name}_{tr_idx}_test_results_{metric}"]["loads"],
                np.mean(y, axis=0),
                label=f"{name}".replace("sp", "OSPF").replace("opt", "Oracle Routing"),
                linestyle="--",
                marker="o",
                linewidth=7,
                markersize=20,
            )
            print(metric, name, np.mean(np.array(y)[:, :-1]))
        # y = all_exps_results[f"opt_0_{metric}"]["data"]
        # ax.plot(all_exps_results[f"sp_0_{metric}"]["loads"], y, label="OPT", linestyle="--", color="green")
        metric_name = (
            f"{''.join(metric.split('_')[1:])}".replace("cost", "Cost")
            .replace("lossrate", "Loss Rate")
            .replace("delay", " Delay")
            .replace("overlay", "Overlay ")
            .replace("global", "Global ")
        )
        plt.xlabel(xlabel="Load Factor")
        plt.xticks(
            ticks=[60, 70, 80, 90, 100, 110, 120],
            labels=["60%", "70%", "80%", "90%", "100%", "110%", "120%"],
        )
        plt.ylabel(ylabel=f"{metric_name}")
        # plt.legend(loc="upper left")
        # plt.title(f"Variation of {metric_name} with load factor in Offband setting")
        # plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/{metric_name}_vs_loads.pdf".replace(" ", "_"),bbox_inches='tight', pad_inches=0.1)


import matplotlib.pyplot as plt

# Additional plots
# 1)------------------------------------------------------------
# %% plot geant topology and tag overlay nodes
import networkx as nx
import numpy as np

# load the topology from adjacency matrix
G = nx.from_numpy_matrix(
    np.loadtxt(
        "/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/topology_files/physical_adjacency_matrix.txt"
    )
)
# nodes_coords = np.loadtxt("/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/topology_files/node_coordinates.txt")
overlay_nodes = np.loadtxt(
    "/home/redha/PRISMA_copy/prisma/examples/overlay_full_mesh_10n_geant/topology_files/map_overlay.txt"
)
nx.draw(
    G,
    node_size=100,
    edge_color="grey",
    width=1,
    with_labels=True,
    font_size=10,
    labels={node: node for node in G.nodes()},
    pos=nx.kamada_kawai_layout(G),
)

# %%
for model in dqn_models:
    y = []
    y_min = []
    y_max = []
    x = []
    for sync_step in sync_steps:
        temp1 = np.mean(
            all_exps_results[
                "dqn_model_sharing"
                + model
                + f"_{sync_step}_constrained_0.9_final_test_results_test_global_cost"
            ]["data"],
            axis=0,
        )
        temp2 = (
            all_exps_results[
                "dqn_model_sharing" + model + f"_{sync_step}_5_overlay_big_signalling_bytes"
            ]["data"]
            + all_exps_results[
                "dqn_model_sharing" + model + f"_{sync_step}_5_overlay_small_signalling_bytes"
            ]["data"]
        ) / all_exps_results[
            "dqn_model_sharing" + model + f"_{sync_step}_5_overlay_data_pkts_injected_bytes_time"
        ][
            "data"
        ]
        y.append(np.mean(temp1))
        x.append(np.mean(temp2))
        y_min.append(np.mean(temp1) - np.std(temp1))
        y_max.append(np.mean(temp1) + np.std(temp1))
    # add the plot into the figure with min and max values shaded area
    ax.plot(x, y, label=f"dqn_model_sharing{model}", marker="o")
    ax.fill_between(x, y_min, y_max, alpha=0.2)

# add digital twin
temp1 = np.mean(
    all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_test_global_cost"]["data"], axis=0
)
temp2 = (
    all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_overlay_big_signalling_bytes"][
        "data"
    ]
    + all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_overlay_small_signalling_bytes"][
        "data"
    ]
) / all_exps_results["dqn_model_sharing" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"][
    "data"
]
ax.plot(
    np.mean(temp2),
    np.mean(temp1),
    label="DT",
    marker="*",
    color="black",
    markersize=10,
)

# add dqn_value_sharing
temp1 = np.mean(
    all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_test_global_cost"]["data"], axis=0
)
temp2 = (
    all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"]
    + all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"]
) / all_exps_results["dqn_model_sharing" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"][
    "data"
]
ax.plot(
    np.mean(temp2),
    np.mean(temp1),
    label="dqn_value_sharing",
    marker="*",
    color="grey",
    markersize=10,
)

# add sp and opt
y = np.mean(all_exps_results["sp_1_5_test_global_cost"]["data"], axis=0)
ax.hlines(np.mean(y), 0, 6, label="SP", linestyle="--", color="red")
y = np.mean(all_exps_results["opt_1_5_test_global_cost"]["data"], axis=0)
ax.hlines(np.mean(y), 0, 6, label="OPT", linestyle="--", color="green")
plt.ylabel(ylabel="Average Cost")
plt.xlabel(xlabel="Overhead Ratio")
plt.tight_layout()
plt.legend()
# plt.savefig(f"{figures_dir}/overhead_ratio_vs_cost.png")
# 5)------------------------------------------------------------
# %% plot the results & save the figures for overhead vs sync step
fig, ax = plt.subplots(figsize=(19.4, 10))
# add dqn buffer dqn_model_sharing model
for signaling_type in signaling_types[:-2]:
    for loss_penalty_type in ["constrained"]:
        for snapshot in snapshots:
            for training_load in [0.9]:
                y = []
                y_min = []
                y_max = []
                x = []
                for sync_step in sync_steps:
                    if signaling_type != "dqn_model_sharing":
                        temp = all_exps_results[
                            f"{signaling_type}_{3}_{loss_penalty_type}_{training_load}_{snapshot}_stats_signalling ratio"
                        ]["data"].max()
                    else:
                        temp = all_exps_results[
                            f"{signaling_type}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results_test_global_cost"
                        ]["data"]
                    y.append(np.mean(temp))
                    y_min.append(np.mean(temp) - np.std(temp))
                    y_max.append(np.mean(temp) + np.std(temp))
                    x.append(sync_step)
                # add the plot into the figure with min and max values shaded area
                overheads[signaling_type] = y
                ax.plot(
                    x,
                    y,
                    label=f"{signaling_type}",
                    marker="o",
                    linewidth=7,
                    markersize=20,
                )
                ax.fill_between(x, y_min, y_max, alpha=0.2)
# add digital twin
# temp = (all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["dqn_logit_sharing" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["dqn_model_sharing" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
# add dqn_value_sharing
# temp = (all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_overlay_big_signalling_bytes"]["data"] + all_exps_results["dqn_value_sharing" + "" + f"_{1}_5_overlay_small_signalling_bytes"]["data"])/all_exps_results["dqn_model_sharing" + "" + f"_{1}_5_overlay_data_pkts_injected_bytes_time"]["data"]
# plt.hlines(np.mean(temp), 1, 8, label=f"dqn_value_sharing", linestyle="--", color="grey")
plt.xlabel(xlabel="Sync step")
plt.ylabel(ylabel="Overhead Ratio")
plt.tight_layout()
plt.legend()
# plt.savefig(f"/home/redha/PRISMA_copy/prisma/figures/final/overhead_ratio_vs_sync_steps.jpeg",bbox_inches='tight', pad_inches=0.1)
# plt.savefig(f"{figures_dir}/overhead_ratio_vs_sync_steps.png")


# %% plot the results & save the figures for average cost, loss and delay over load factors vs episodes for each sync step

figures_dir = "/home/redha/PRISMA_copy/prisma/figures"
sync_step = 1
threshold = 0.0
# snapshots.remove("episode_13_step_13")
for metric in metric_names["test_results"]:
    fig, ax = plt.subplots(figsize=(19.4, 10))
    # add dqn buffer dqn_model_sharing model
    for training_load in [
        0.9,
    ]:
        for loss_penalty_type in loss_penalty_types:
            for model in dqn_models:
                y = []
                y_min = []
                y_max = []
                x = []
                for idx, snapshot in enumerate(snapshots):
                    name = f"dqn_model_sharing{model}_{sync_step}_{loss_penalty_type}_{training_load}_{snapshot}_test_results_test_results"
                    temp = np.mean(
                        all_exps_results[f"{name}_{metric}_{threshold}"]["data"], axis=0
                    )
                    y.append(np.mean(temp))
                    y_min.append(np.mean(temp) - np.std(temp))
                    y_max.append(np.mean(temp) + np.std(temp))
                    x.append(idx)
                # add the plot into the figure with min and max values shaded area
                ax.plot(
                    x,
                    y,
                    label=f"{loss_penalty_type}_{training_load}_{threshold}".replace(
                        "None", "Loss Blind"
                    )
                    .replace("fixed", "Loss Aware")
                    .replace("constrained", "RCPO"),
                    marker=".",
                )
                # ax.fill_between(x, y_min, y_max, alpha=0.2)
    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_1_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 1", linestyle="--", color="blue")
    # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_5_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT", linestyle="--")
    # # add digital twin
    # temp = np.mean(all_exps_results[f"digital_twin_1_10_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label="DT 10", linestyle="--")
    # add dqn_value_sharing
    # temp = np.mean(all_exps_results[f"target_1_5_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(temp), 1, 8, label=f"dqn_value_sharing", linestyle="--", color="grey")

    # add sp and opt
    y = np.mean(all_exps_results[f"sp_0_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(y), 1, 5, label="SP", linestyle="--", color="red")
    # y = np.mean(all_exps_results[f"opt_0_{metric}"]["data"], axis=0)
    # ax.hlines(np.mean(y), 1, 5, label="OPT", linestyle="--", color="green")

    plt.xlabel(xlabel="Episode")
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.title(f"{metric} over loads for sync step {sync_step}")
    # plt.savefig(f"{figures_dir}/{''.join(metric.split('_')[-2:])}_vs_sync_steps_new.png")


import sys

# %%plot hist of gaps for sync step 1 mat 0 model sharing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("/home/redha/PRISMA_copy/prisma/source/")
from utils import convert_tb_data

path = "/home/redha/PRISMA_copy/prisma/examples/5n_overlay_full_mesh_abilene/results/test/NN_threshold_0.0_fixed_0.9_sync_1/estimation"
names = ["neighbor_estimation", "node_estimation", "gap"]

results = {}
# load the results
exp_df = convert_tb_data(path)
# get the values
for name in names:
    results[name] = exp_df[exp_df["name"] == name]["value"].values


plot_style = {
    "font.family": "Times New Roman",
    "font.size": 50,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 3,
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.facecolor": "#EBEBEB",
    "xtick.color": "black",
    "ytick.color": "black",
    "ytick.minor.visible": False,
    "xtick.minor.visible": False,
    "grid.color": "black",
    "grid.linestyle": "-.",
    "grid.linewidth": 0.6,
    "figure.figsize": (19.4, 10),
}
max_limit_gap = 200
plt.rcParams.update(plot_style)
fig, ax = plt.subplots(figsize=(19.4, 10))
percent_gap = results["gap"] / results["node_estimation"]
# percent_gap = np.random.randn(1000)
plt.hist(
    percent_gap * 100,
    bins=max(len(percent_gap) // 30, 30),
    label="Gap",
    alpha=0.5,
    color="blue",
    density=True,
    range=(0, max_limit_gap),
)
plt.xlabel(xlabel="Normalized TD error (%)")
plt.xticks(np.arange(0, max_limit_gap, 10), rotation=45)
# rotate xticks

plt.ylabel(ylabel="Frequency")
plt.tight_layout()
# plt.legend(loc="upper right")
plt.title("Normalized TD error distribution with no filtering")

xs = np.arange(0, max_limit_gap, 0.005)
xs = np.arange(0, max_limit_gap, 10)
counts = []
for x in xs:
    count = np.sum(
        np.logical_or(
            results["neighbor_estimation"] < 0,
            np.logical_and(
                np.logical_and(
                    results["node_estimation"] > 0, results["neighbor_estimation"] > 0
                ),
                (results["gap"] / results["node_estimation"]) * 100 > x,
            ),
        )
    )
    counts.append(count)
    print(
        "len",
        len(percent_gap),
        "filtred len",
        len(percent_gap[percent_gap > x]),
        "count",
        count,
        "ratio",
        (len(percent_gap) / np.array(count)),
        "x",
        np.round(x, 3),
    )
fig, ax = plt.subplots(figsize=(19.4, 10))
y = (np.array(counts) / len(percent_gap)) * 0.31
plt.plot(xs, y, label="Gap > x", linewidth=7)
# plt.xticks(np.arange(0, 3.2, 0.25), rotation=45)
plt.xticks(xs, rotation=45)
plt.yticks(np.arange(0, 0.32, 0.05))
plt.xlabel(xlabel="Threshold (%)")
plt.ylabel(ylabel="Signalling Overhead")
plt.title("Signalling overhead vs threshold (estimation)")
plt.tight_layout()

import sys

# %%plot hist of gaps for sync step 1 mat 0 model sharing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("/home/redha/PRISMA_copy/prisma/source/")
from utils import convert_tb_data

thresholds = [0.0, 0.12, 0.25, 0.5, 0.75, 1.0]
overheads = []
ping_freq = 0.75
costs = []
for threshold in thresholds:
    path = f"/home/redha/PRISMA_copy/prisma/examples/5n_overlay_full_mesh_abilene/results/test_1/NN_threshold_{threshold}_fixed_0.9_sync_1_ping_{ping_freq}/stats"
    # load the results
    exp_df = convert_tb_data(path)
    overheads.append(exp_df[exp_df["name"] == "signalling ratio"]["value"].values[-1])

    path = f"/home/redha/PRISMA_copy/prisma/examples/5n_overlay_full_mesh_abilene/results/test_1/NN_threshold_{threshold}_fixed_0.9_sync_1_ping_{ping_freq}/test_results"

    # load the results
    exp_df = convert_tb_data(path)
    costs.append(np.mean(exp_df[exp_df["name"] == "test_global_cost"]["value"]))


plot_style = {
    "font.family": "Times New Roman",
    "font.size": 50,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 3,
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "axes.facecolor": "#EBEBEB",
    "xtick.color": "black",
    "ytick.color": "black",
    "ytick.minor.visible": False,
    "xtick.minor.visible": False,
    "grid.color": "black",
    "grid.linestyle": "-.",
    "grid.linewidth": 0.6,
    "figure.figsize": (19.4, 10),
}


fig, ax = plt.subplots(figsize=(19.4, 10))

plt.plot(thresholds, overheads, label="Gap > x", linewidth=7)
# plt.xticks(np.arange(0, 3.2, 0.25), rotation=45)
plt.xticks(thresholds, np.array(thresholds) * 100, rotation=45)
plt.yticks(np.arange(min(overheads), max(overheads), 0.05))
plt.xlabel(xlabel="Threshold (%)")
plt.ylabel(ylabel="Signalling Overhead")
plt.title("Signalling overhead vs threshold (measured)")
plt.tight_layout()
fig, ax = plt.subplots(figsize=(19.4, 10))

plt.plot(thresholds, costs, label="Gap > x", linewidth=7)
# plt.xticks(np.arange(0, 3.2, 0.25), rotation=45)
plt.xticks(thresholds, np.array(thresholds) * 100, rotation=45)
plt.yticks(np.arange(min(costs), max(costs), 0.1))
plt.xlabel(xlabel="Threshold (%)")
plt.ylabel(ylabel="Cost")
plt.title("Cost vs threshold (measured)")
plt.tight_layout()


# %% plot the results & save the figures for cost, loss and delay vs load factors
# key = f"{signaling_type}{dqn_model}_{sync_step}_{loss_pen_type}_{lambda_wait}_{buffer_soft_limit}_{traff_mat}_{metric_name}"
model_key = [
    "NN_1_constrained_0_0.1_final",
    "NN_1_constrained_0_0.4_final",
    "NN_1_constrained_0_0.6_final",
    "NN_1_constrained_25_0.1_final",
    "NN_1_constrained_25_0.4_final",
    "NN_1_constrained_25_0.6_final",
    "NN_1_constrained_60_0.1_final",
    "NN_1_fixed_0_0.6_final",
]
model_names = [
    "RCPO after 0s 10%",
    "RCPO after 0s 40%",
    "RCPO after 0s 60%",
    "RCPO after 25s 10%",
    "RCPO after 25s 40%",
    "RCPO after 25s 60%",
    "loss blind",
    "Loss penalty",
]
mat = 0
for metric in metric_names["test_results"][:1]:
    # add dqn buffer dqn_model_sharing model
    fig, ax = plt.subplots(figsize=(19.4, 10))
    for i in range(len(model_names)):
        y = []
        temp = []
        y_min = []
        y_max = []
        x = []
        # for sync_step in sync_steps:
        for mat in [
            0,
        ]:
            if len(temp) == 0:
                temp = all_exps_results[model_key[i] + f"_{mat}_{metric}"]["data"][:]

            else:
                temp = np.concatenate(
                    (temp, all_exps_results[model_key[i] + f"_{mat}_{metric}"]["data"]),
                    axis=1,
                )
        y = np.mean(temp, axis=1)
        print(f"{model_names[i]}", np.mean(y))
        y_min = np.min(temp, axis=1)
        y_max = np.max(temp, axis=1)
        x = all_exps_results[model_key[i] + f"_{mat}_{metric}"]["loads"]
        # add the plot into the figure with min and max values shaded area
        ax.plot(x, y, label=f"{model_names[i]}", marker="o")
        # ax.fill_between(x, y_min, y_max, alpha=0.2)

    plt.xlabel(xlabel="Load Factor")
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"{figures_dir}/{''.join(metric.split('_')[-2:])}_vs_sync_steps_new.png")

# %% plot the results for the average cost vs epochs
model_key = [
    # "NN_1_constrained_0_0.1_",
    "NN_1_constrained_0_0.4_",
    # "NN_1_constrained_0_0.6_",
    # "NN_1_constrained_25_0.1_",
    "NN_1_constrained_25_0.4_",
    # "NN_1_constrained_25_0.6_",
    "NN_1_constrained_60_0.1_",
    "NN_1_fixed_0_0.6_",
    "sp",
    "opt",
]
model_names = [
    #    "RCPO after 0s 10%",
    "RCPO after 0s 40%",
    #    "RCPO after 0s 60%",
    #     "RCPO after 25s 10%",
    "RCPO after 25s 40%",
    #    "RCPO after 25s 60%",
    "loss blind",
    "Loss penalty",
    "SP",
    "OPT",
]
snapshots = snapshots[:-1]
fig, ax = plt.subplots(figsize=(19.4, 10))
for i in range(len(model_key)):
    y = []
    for snapshot in snapshots:
        if model_key[i] in ("sp", "opt"):
            snapshot = ""
        for metric in metric_names["test_results"][:1]:
            temp = []
            y_min = []
            y_max = []
            x = []
            # for sync_step in sync_steps:
            for mat in [
                0,
            ]:
                if len(temp) == 0:
                    temp = all_exps_results[
                        model_key[i] + f"{snapshot}_{mat}_{metric}"
                    ]["data"][:]

                else:
                    temp = np.concatenate(
                        (
                            temp,
                            all_exps_results[
                                model_key[i] + f"{snapshot}_{mat}_{metric}"
                            ]["data"],
                        ),
                        axis=1,
                    )
            # y = np.mean(temp, axis=1)
            print(f"{model_names[i]}", np.mean(temp))
            y.append(np.mean(temp))
            y_min = np.min(temp, axis=1)
            y_max = np.max(temp, axis=1)
            x = all_exps_results[model_key[i] + f"{snapshot}_{mat}_{metric}"]["loads"]
            # add the plot into the figure with min and max values shaded area
    ax.plot(range(len(snapshots)), y, label=f"{model_names[i]}", marker="o")
    # ax.fill_between(x, y_min, y_max, alpha=0.2)

    plt.xlabel(xlabel="Snapshot Number")
    plt.xticks(
        range(len(snapshots)), [" ".join(val.split("_")[-1:]) for val in snapshots]
    )
    plt.ylabel(ylabel=f"{''.join(metric.split('_')[-2:])}")
    plt.tight_layout()
    # plt.title(f"{model_names[i]}")
    plt.legend()
# %%
