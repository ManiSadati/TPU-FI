import numpy as np
from utils import attention_calculation
import os

attention_layers, map_attention_layer = attention_calculation()



files = set(os.listdir("diff_results"))
for analysis in ["sum", "num_diff"]:
    for fi_type in ['single', 'small-box', 'medium-box']:
        result_start = [[{} for j in range(8)]for i in range(3)]
        count_start = [[{} for j in range(8)]for i in range(3)]

        result_end = [[{} for j in range(8)]for i in range(4)]
        count_end = [[{} for j in range(8)]for i in range(4)]
        with open(f"./attention_results/{fi_type}-{analysis}-results.txt", "w") as output_file:
            for i in range(191):
                for filename in files:
                    if filename.endswith(".npy") and i == int(filename.split("-")[1]):
                        type = filename.split("-")[3:-1]
                        if len(type) == 2:
                            type[0] += "-" + type[1]
                        type = type[0]
                        if type != fi_type:
                            continue
                        output_layer = int(filename.split("-")[0].split("_")[1])
                        start_layer = int(filename.split("-")[1])
                        s_block, s_head, s_l = map_attention_layer[start_layer]
                        diff = np.load(f"diff_results/{filename}")
                        if output_layer in map_attention_layer:
                            o_block, o_head, o_l = map_attention_layer[output_layer]
                        else:
                            o_block, o_head, o_l = 3, 0, 0
                        if (o_block, o_head) not in result_start[s_block][s_head].keys():
                            result_start[s_block][s_head][(o_block, o_head)] = 0
                            count_start[s_block][s_head][(o_block, o_head)] = 0
                        
                        if analysis == "sum":
                            result_start[s_block][s_head][(o_block, o_head)] += np.abs(diff).sum().item()
                        else:
                            result_start[s_block][s_head][(o_block, o_head)] += np.abs(diff!=0.).sum().item()
                        count_start[s_block][s_head][(o_block, o_head)] += 1

                        if o_block == s_block:
                            continue

                        if (s_block, s_head) not in result_end[o_block][o_head].keys():
                            result_end[o_block][o_head][(s_block, s_head)] = 0
                            count_end[o_block][o_head][(s_block, s_head)] = 0
                        
                        if analysis == "sum":
                            result_end[o_block][o_head][(s_block, s_head)] += np.abs(diff).sum().item()
                        else:
                            result_end[o_block][o_head][(s_block, s_head)] += np.abs(diff!=0.).sum().item()
                        
                        count_end[o_block][o_head][(s_block, s_head)] += 1

            output_file.write("\nInjection Results, based on where injection Initiated:\n")
            for i in range(3):
                for j in range(8):
                    total = 0.
                    for k in result_start[i][j].keys():
                        total += int(result_start[i][j][k] / count_start[i][j][k])
                    output_file.write(f"\n\nFI [{i},{j}] = {total}:\n")
                    for b, h in sorted(result_start[i][j].keys(), key=lambda k: result_start[i][j][k]):
                        output_file.write(f"[{b},{h}]={int(result_start[i][j][(b,h)] / count_start[i][j][(b,h)])} ")
                    output_file.write(" ")

            output_file.write("\n-----------\n")

            output_file.write("\nInjection Results, based on where injection was observed:\n")
            for i in range(1,4):
                for j in range(8):
                    if(i == 3 and j):
                        continue
                    total = 0.
                    for k in result_end[i][j].keys():
                        total += int(result_end[i][j][k] / count_end[i][j][k])
                    output_file.write(f"\n\nFI [{i},{j}] = {total}:\n")
                    for b, h in sorted(result_end[i][j].keys(), key=lambda k: result_end[i][j][k]):
                        output_file.write(f"[{b},{h}]={int(result_end[i][j][(b,h)] / count_end[i][j][(b,h)])} ")
                    output_file.write(" ")

