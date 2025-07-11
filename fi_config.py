import random
import math
import numpy as np
import os, shutil

def init_fi():
    shutil.rmtree("./diff_results", ignore_errors=True)
    os.makedirs("./diff_results", exist_ok=True)


def reset_fi_folder():
    shutil.rmtree("./fi", ignore_errors=True)
    os.makedirs("./fi", exist_ok=True)
    
def reset_files():
    open("./fi/layer_num.txt", "w").close()
    open("./fi/mode.txt", "w").close()
    open("./fi/locations.txt", "w").close()
    open("./fi/dimension.txt", "w").close()
    open("./fi/layer_output_list.txt", "w").close()

def fi_init_profile(layer, img_index, layer_output_list):
    reset_files()
    with open("./fi/mode.txt", "w") as file:
        # profiling layer, img, fault_type (None), iteration (-1)
        file.write(f"profiling {layer} {img_index} None -1\n")
    with open("./fi/layer_output_list.txt", "w") as file:
        for layer_output in layer_output_list:
            file.write(f"{layer_output}\n")

def get_dims():
    with open("./fi/dimension.txt", "r") as file:
        dimensions = file.read().split()
    return dimensions


def fi_init_inject(layer, img_index, type, it, dimensions):
    # Define initial values for variables
    box_x = box_y = l_x = r_x = l_y = r_y = -1
    x_size = y_size = c_size = 0
    num_ops = 0
    prob = 0.0

    # Writing initial layer number
    with open("./fi/layer_num.txt", "w") as file:
        file.write("0\n")

    # Reading dimensions
    # with open("./fi/dimension.txt", "r") as file:
    #     dimensions = file.read().split()
    #     print(dimensions)
    layer_name = dimensions[0]
    c_size = int(dimensions[2])
    x_size = int(dimensions[3])
    y_size = int(dimensions[4])
    num_ops = int(dimensions[5])
    

    # Determine the type and calculate probabilities and dimensions
    if type == "single":
        box_x = box_y = 1
        prob = 1.0
    elif type == "small-box":
        area = random.randint(41, max(41,min(x_size * y_size,113)))
        if area > x_size * y_size:
            return layer_name, -1, c_size,  x_size * y_size, num_ops
        box_y = random.randint(max(1,math.ceil(area/x_size)), min(y_size, area))
        box_x = max(1, area // box_y)
        if box_x > x_size:
            return layer_name, -1, c_size,  x_size * y_size, num_ops
        prob = 0.07
    elif type == "medium-box":
        area = random.randint(949, max(949,min(x_size * y_size,1351)))
        print ("area: ",area)
        if area > x_size * y_size:
            return layer_name, -1, c_size,  x_size * y_size, num_ops
        box_y = random.randint(max(1,math.ceil(area/x_size)), min(y_size, area))
        box_x = max(1, area // box_y)
        print ("box_x , x_size, box_y, y_size ",box_x, x_size, box_y, y_size)
        if box_x > x_size:
            return layer_name, -1, c_size,  x_size * y_size, num_ops
        prob = 0.03

    l_x = random.randint(0, x_size - box_x)
    l_y = random.randint(0, y_size - box_y)
    l_c = random.randint(0, c_size - 1)

    # Create a random matrix and apply probability threshold
    matrix = np.random.rand(box_x, box_y) < prob

    # Extract locations where the condition is True
    locs = np.argwhere(matrix)

    # Write to fi_locations.txt based on calculated faults
    with open("./fi/locations.txt", "w") as file:
        for loc in locs:
            fi_bit = 0 if random.random() <= 0.59 else random.randint(1, 7) # do +-1 with prob of 59% and other bitflips otherwise.
            file.write(f"{l_c} {loc[0] + l_x} {loc[1] + l_y} {fi_bit}\n")

    # Update mode file with layer and mode
    with open("./fi/mode.txt", "w") as file:
        file.write(f"injection {layer} {img_index} {type} {it}\n")
    
    return layer_name, 0, c_size, x_size * y_size, num_ops

def get_tensor_from_file(layer_output, fi_layer, img_index, type, it):
    file_path = f"./fi/output_{layer_output}-{fi_layer}-{img_index}-{type}-{it}.txt"
    if not os.path.exists(file_path):
        print("path not exists",(file_path))
        return None, -1
    with open(file_path, "r") as file:
        # read first line
        c_size, x_size, y_size = map(int, file.readline().split())
        output = file.read().splitlines()
        output = np.array([list(map(float, line.split())) for line in output])
        output = output.reshape((c_size, x_size, y_size))
    return output, 1



def fi_post_process(layer_output_list, fi_layer, img_index, fault_types, max_iterations):
    for layer_output in layer_output_list:
        golden_tensor, _ = get_tensor_from_file(layer_output, fi_layer, img_index, "None", -1)
        if _ == -1:
            print("Golden tensor not found!")
            exit()
        for type in fault_types:
            for it in range(max_iterations):
                output_tensor, status = get_tensor_from_file(layer_output, fi_layer, img_index, type, it)
                if status == -1:
                    continue
                diff = output_tensor - golden_tensor
                np.save(f"./diff_results/diff_{layer_output}-{fi_layer}-{img_index}-{type}-{it}.npy", diff)
    return