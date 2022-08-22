# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:16:40 2022

@author: Redha

Plot for ideal, nn and target, the delay, loss and cost over load for each sync step
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.size': 60})
plt.rcParams.update({'font.weight': 'bold'})
# plt.rcParams.update({'font.serif': ["Times"]})
if __name__ == '__main__':
    ### define params
    stat_names = ["Average Cost Per Packets",
                  "Avg e2e Delay", 
                  "Avg loss rate"]
    stat_file_names = ["loss", "delay", "rew", "real_rew", "total_rew", "overhead"]
    folder_path = "."
    
    kkk = 3
    """ 1 ==  sync step variation for ideal case 
        2 == signaling type comparaison for sync step 1s and 20% ratio 
        3 == ratio comparaison  for NN signaling 
    """
    model_names = ["DQN Routing - Model Sharing", "Shortest Path Routing"]
    folder_name =  f"tests_sync_variation_mat0_rb_10k_plots"
    line_styles = ["solid" , "solid", "dashed", "dashed"]
    colors = [ "purple", "green", "red","blue"]
    avg_data = {}
    avg_data_loads = {}
    test_folder = "_tests_overlay_6"
    signaling_inband=1
    rb_size = 10000
    lite = ("", "","","", "")
    ### check if folder exists
    if folder_name not in os.listdir(folder_path):
        os.mkdir(folder_path + "/" + folder_name)
    for signaling_inband in (1,):
        for kkk_idx, kkk in enumerate((12, 11, 7)):
            # fig1, ax1 = plt.subplots()
            # fig1.set_size_inches(19.4, 10)
            # [0.6428656100517803, 0.47653319732663824, 0.44059512730308725, 0.36639229472221074, 0.5688626373310568, ]
            for xxx, signaling_type in enumerate(["NN",  "sp", "target", "prio"]):
                #if(xxx==1):
                #    continue
                # if kkk_idx==1 and xxx >1:
                    # continue
                for stat_idx in range(len(stat_names)):
                    if stat_idx != kkk_idx:
                        continue
                    ## stat over sync step
                    # fig1 = plt.figure()
                    # fig1.set_size_inches(19.4, 10)
                sync_steps = []
                sync_charge_steps = []
                    
                    
                    # fig = plt.figure()
                    # fig.set_size_inches(19.4, 10)
                #list_charges = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200, 250, 300]
                list_charges = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
                    # list_charges = [40]
            
            
                syncs = [1000]#np.arange(1000, 8000, 1000).tolist()
                overlayPackets = [20]#,20,50,100]#[5,10,20,50,100,500]
                    # syncs.remove(6000)
                    # syncs.remove(7000)
                    # syncs = np.arange(500, 5000, 500).tolist() + np.arange(5000, 15000, 5000).tolist()
                if signaling_type in ("ideal", "NN"):
                    names = [f"prio_0_dqn_buffer{lite[xxx]}_NN_{signaling_inband}_fixed_rb_10000_sync{xx}ms_ratio_10_overlayPackets_{yy}" for xx in syncs for yy in overlayPackets] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy}" for xx in np.array(syncs)/1000 for yy in overlayPackets] 
                elif signaling_type in ("target"):
                    print("here")
                    names = [f"prio_0_dqn_buffer{lite[xxx]}_target_{signaling_inband}_fixed_rb_10000_sync{xx}ms_ratio_10_overlayPackets_{yy}" for xx in syncs for yy in overlayPackets] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy}" for xx in np.array(syncs)/1000 for yy in overlayPackets]  
                elif signaling_type in ("prio"):
                    names = [f"prio_1_dqn_buffer{lite[xxx]}_target_{signaling_inband}_fixed_rb_10000_sync{xx}ms_ratio_10_overlayPackets_{yy}" for xx in syncs for yy in overlayPackets] 
                    official_names = [f"DQN Buffer {signaling_type} sync {xx}s overlayPackets {yy}" for xx in np.array(syncs)/1000 for yy in overlayPackets]
                
                elif signaling_type == "sp":
                    names = ["prio_sp_ideal_1_fixed_rb_10000_sync1000ms_ratio_10_overlayPackets_50"]* len(syncs)
                    official_names = ["Shortest Path"]* len(syncs)
                #else:
                #    names = ["opt_ideal_0_fixed_rb_20000_sync10000ms_ratio_10"] * len(syncs) 
                #    official_names = ["Optimal Solution"] * len(syncs)
                j = 0
                for i in range(len(names)):
                    for charge_index, charge in enumerate(list_charges):
                        #if (signaling_type == "sp" or signaling_type == "opt") and charge == 40:
                        #    continue
                        temp = np.loadtxt(f"{test_folder}/{names[i]}_load_{charge}.txt", delimiter=',', dtype=object)
                        #print(names[i], temp.shape)
                        if len(temp.shape) == 1:
                            delay_temp = np.array(temp[kkk], dtype=float).reshape(1, -1)
                        else:
                            delay_temp = np.array(temp[:, kkk], dtype=float).mean(axis=0).reshape(1, -1)
                
                        if charge_index == 0:
                            delay = delay_temp.reshape(1, -1)
                        else:
                            delay = np.concatenate((delay, delay_temp.reshape(1, -1)))
                        
                    sync_steps.append(np.mean(delay, axis=0))
                    sync_charge_steps.append(delay)
    
                    j += 1
                print(stat_names[kkk_idx], sync_steps)
                    
                best_sync = np.argmin(np.array(sync_steps), axis=0)
                print(best_sync)
                print(sync_charge_steps[best_sync[0]])
                avg_data_loads[f"{signaling_type}{lite[xxx]}_{stat_names[kkk_idx]}_{signaling_inband}_{test_folder}"] = np.array(sync_charge_steps[best_sync[0]])[:, 0]
                avg_data[f"{signaling_type}{lite[xxx]}_{stat_names[kkk_idx]}_{signaling_inband}_{test_folder}"] = np.array(sync_steps)[:, 0]
                        
                #ax1.plot(np.array(syncs)/1000, avg_data[f"{signaling_type}_{kkk}_{signaling_inband}"] , label=f"{model_names[xxx]}", color=colors[xxx], linestyle=line_styles[xxx], marker="o")
        
                
            # plt.vlines(min_x_value, 0, 3.5, linestyles="dotted", label="Minimum value")
            # ax1.set_ylabel(f"{stat_names[kkk]}")
            # ax1.set_xlabel(f"Synchronisation Period T_s (s)")
            # ax1.set_xlim(0, 21)
            # ax1.set_ylim(0, 3.5)
            # ax1.set_xticks(np.arange(0, 25, 5).tolist() + [min_x_value], np.arange(0, 25, 5).tolist() + [min_x_value] )
            
            # ax2 = ax1.twinx()
            # ax2.set_yscale('linear')
            # mse = (avg_data["ideal"] - avg_data["NN"])**2
            # ax2.plot(np.array(syncs)/1000, mse, color="green", linestyle=line_styles[0], marker="o", label="MSE Between No Signaling and Off-band Signaling")
            # ax2.set_ylim(0, np.max(mse))
            # ax2.set_ylabel(f"Mean Squared Error", color="green")
            # ax2.tick_params(axis='y', labelcolor="green")
        
            # fig1.legend(loc=2)
            # fig1.tight_layout()

            
        
            fig2, ax2 = plt.subplots()
            fig2.set_size_inches(19.4, 10)
            ax2.plot(np.array(list_charges)/100,
                 avg_data_loads[f"NN_{stat_names[kkk_idx]}_1_{test_folder}"],
                  label=f"Model sharing", linestyle=line_styles[0],
                  marker="o",
                  color=colors[0],
                  linewidth=7,
                  markersize=20)
            ax2.plot(np.array(list_charges)/100,
                 avg_data_loads[f"target_{stat_names[kkk_idx]}_1_{test_folder}"],
                  label=f"Value sharing", linestyle=line_styles[0],
                  marker="o",
                  color=colors[1],
                  linewidth=7,
                  markersize=20)
            ax2.plot(np.array(list_charges)/100,
                 avg_data_loads[f"sp_{stat_names[kkk_idx]}_1_{test_folder}"],
                  label=f"SP", linestyle=line_styles[2],
                  marker="o",
                  color=colors[2],
                  linewidth=7,
                  markersize=20)
            ax2.plot(np.array(list_charges)/100,
                 avg_data_loads[f"prio_{stat_names[kkk_idx]}_1_{test_folder}"],
                  label=f"Prio Value sharing", linestyle=line_styles[0],
                  marker="o",
                  color=colors[3],
                  linewidth=7,
                  markersize=20)
            #if(kkk_idx==0): ax2.set_ylim(0.0, 2.0)
            #if(kkk_idx==1): ax2.set_ylim(30, 500)
            #if(kkk_idx==2): ax2.set_ylim(0.0, 0.5)
            ax2.set_xlim(0.6, 2.4)
            ax2.set_xticks(np.array(list_charges)/100, np.array(list_charges)/100)#(np.arange(1, 9, 1, dtype =int), np.arange(1, 9, 1, dtype =int))
            fig2.legend(prop={'weight':'normal'})
            fig2.tight_layout()
            ax2.set_xlabel(f"Load charge ", fontweight="bold")
            ax2.set_ylabel(f"{stat_names[kkk_idx]}", fontweight="bold")
            plt.savefig(f"pictures/avg_{stat_names[kkk_idx]}_overlay_load_inband.png")
            plt.show()
    #fig2, ax2 = plt.subplots()
    #fig2.set_size_inches(19.4, 10)
    #ax2.plot( avg_data['NN_Singalling overhead_1__train_final']/avg_data["NN_data_1__train_final"],
    #         avg_data["NN_Average Cost Per Packets_1__tests_final"],
    #          label=f"nn", linestyle=line_styles[0], marker="o")
    
    # ax2.plot( avg_data['NN_lite_Singalling overhead_1__train_final']/avg_data["NN_lite_data_1__train_final"],
    #          avg_data["NN_lite_Average Cost Per Packets_1__tests_final"],
    #           label=f"nn lite", linestyle=line_styles[0], marker="o")
    
    #fig2.legend(loc=2, prop={'weight':'normal'})
    #fig2.tight_layout()
    #ax2.set_xlabel(f"Signaling Overhead (%) ", fontweight="bold")
    #ax2.set_ylabel(f"Average Cost per packet", fontweight="bold")
    
    ## plot cost vs sync steps
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(19.4, 10)
    ax2.plot(overlayPackets,
             avg_data[f"NN_Average Cost Per Packets_1_{test_folder}"],
              label=f"Model sharing", linestyle=line_styles[0],
              marker="o",
              color=colors[0],
              linewidth=7,
              markersize=20)
    ax2.plot(overlayPackets,
             avg_data[f"target_Average Cost Per Packets_1_{test_folder}"],
              label=f"Value sharing", linestyle=line_styles[0],
              marker="o",
              color=colors[1],
              linewidth=7,
              markersize=20)
    ax2.plot(overlayPackets,
             avg_data[f"prio_Average Cost Per Packets_1_{test_folder}"],
              label=f"Prio Value sharing", linestyle=line_styles[0],
              marker="o",
              color=colors[3],
              linewidth=7,
              markersize=20)
    
    
    # ax2.plot( np.array(syncs)/1000,
    #          avg_data["NN_lite_Average Cost Per Packets_1__tests_final"],
    #           label=f"nn lite", linestyle=line_styles[0],
    #           marker="o",
    #           linewidth=7,
    #           markersize=20)
    #ax2.hlines(avg_data["target_Average Cost Per Packets_1__tests_final"],
    #            0 ,
    #            12,
    #            label=f"Value sharing",
    #            color="green",
    #            linestyle="dashed",
    #            linewidth=7)
    #ax2.hlines(avg_data["prio_Average Cost Per Packets_1__tests_final"],
    #            0 ,
    #            12,
    #            label=f"Prioritize Value sharing",
    #            color="black",
    #            linestyle="dashed",
    #            linewidth=7)
    ax2.hlines(avg_data[f"sp_Average Cost Per Packets_1_{test_folder}"],
                0 ,
                500,
                label=f"SP",
                color="red",
                linestyle="dashed",
                linewidth=7)
    
    #ax2.hlines(avg_data["opt_Average Cost Per Packets_1__tests_final"],
    #            0 ,
    #            12,
    #            label=f"Oracle",
    #            color="blue",
    #            linestyle="dashed",
    #            linewidth=7)
    #ax2.set_ylim(0.0, 0.3)
    #ax2.set_xlim(0, 500)
    ax2.set_xticks(np.array(overlayPackets), np.array(overlayPackets))#(np.arange(1, 9, 1, dtype =int), np.arange(1, 9, 1, dtype =int))
    fig2.legend(prop={'weight':'normal'})
    fig2.tight_layout()
    ax2.set_xlabel(f"Overlay Resfreshing Raate ", fontweight="bold")
    ax2.set_ylabel(f"Average Cost per packet", fontweight="bold")
    
    
    ## plot cost vs sync steps
    #fig2, ax2 = plt.subplots()
    #fig2.set_size_inches(19.4, 10)
    #ax2.plot(np.array(syncs)/1000,
    #         avg_data['NN_Singalling overhead_1__train_final']/avg_data["NN_data_1__train_final"],
    #          label=f"nn", linestyle=line_styles[0], marker="o")
    #
    ## ax2.plot( np.array(syncs)/1000,
    ##          avg_data['NN_lite_Singalling overhead_1__train_final']/avg_data["NN_data_1__train_final"],
    ##           label=f"nn lite", linestyle=line_styles[0], marker="o")
    #
    #fig2.legend(prop={'weight':'normal'})
    #fig2.tight_layout()
#
    #ax2.set_xlabel(f"Sync steps " ,fontweight="bold")
    #ax2.set_ylabel(f"Signaling Overhead (%) ", fontweight="bold")
#
#
    #print(avg_data["target_Average Cost Per Packets_1__tests_final"][0])



    # avg_data["NN_overhead_1_train_3"] = avg_data["NN_overhead_1_train_3"] + ((61/(np.array(syncs)/1000))*8057152)
    
    # for i in range(2):
    #     for j in range(2):
    plt.savefig(f"pictures/avg_cost_overlay_refresh_rate_variation_inband.png")
    plt.show()