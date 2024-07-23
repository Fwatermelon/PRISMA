base_folder = "/home/redha/.vscode-server/data/User/History/"

# read all entries.json in the subfolders of subfolders of base_folder and print the keys 

import os, json, time
res = {}
folder_path = "/home/redha/prisma_chap4/prisma/examples/"
for folder in os.listdir(base_folder):
    for file in os.listdir(os.path.join(base_folder, folder)):
        if file == "entries.json":
            with open(os.path.join(base_folder, folder, file)) as f:
                data = json.load(f)
                if folder_path in data["resource"]:
                    # get the most recent entry
                    if data["resource"] not in res:
                        res[data["resource"].split("/")[-1]] = (data["entries"][-1], os.path.join(base_folder, folder), time.ctime(data["entries"][-1]["timestamp"]/1000))
                    for entry in data["entries"]:
                        if entry["timestamp"] > res[data["resource"].split("/")[-1]][0]["timestamp"]:
                            res[data["resource"].split("/")[-1]] = (entry, os.path.join(base_folder, folder), time.ctime(entry["timestamp"]/1000))
# print(res)
for key in res:
    # print(f"{key} : {res[key][1]} : {res[key][2]}")
    print(key, folder)
    print("\n\n")
    # print("copying from ", os.path.join(res[key][1], res[key][0]["id"]), "to", os.path.join(folder_path, key))
    # os.system(f"cp {os.path.join(res[key][1], res[key][0]['id'])} {os.path.join(folder_path, key)}")
               