import random
import numpy as np

def reset_files():
    open("./fi_layer_num.txt", "w").close()
    open("./fi_mode.txt", "w").close()
    open("./fi_locations.txt", "w").close()
    open("./fi_dimension.txt", "w").close()

def fi_init_profile(layer):
    reset_files()
    with open("./fi_mode.txt", "w") as file:
        file.write(f"profiling {layer}\n")


def fi_init_inject(layer, type):
    # Define initial values for variables
    box_x = box_y = l_x = r_x = l_y = r_y = -1
    x_size = y_size = c_size = 0
    num_ops = 0
    prob = 0.0

    # Writing initial layer number
    with open("./fi_layer_num.txt", "w") as file:
        file.write("0\n")

    # Reading dimensions
    with open("./fi_dimension.txt", "r") as file:
        dimensions = file.read().split()
        print(dimensions)
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
        area = random.randint(41, 113)
        if area > x_size * y_size:
            return layer_name, -1,  x_size * y_size, num_ops
        box_y = random.randint(max(1,area//x_size), min(y_size, area))
        box_x = max(1, area // box_y)
        if box_x > x_size:
            return layer_name, -1,  x_size * y_size, num_ops
        prob = 0.07
    elif type == "medium-box":
        area = random.randint(949, 1351)
        if area > x_size * y_size:
            return layer_name, -1,  x_size * y_size, num_ops
        box_y = random.randint(max(1,area//x_size), min(y_size, area))
        box_x = max(1, area // box_y)
        if box_x > x_size:
            return layer_name, -1,  x_size * y_size, num_ops
        prob = 0.03

    l_x = random.randint(0, x_size - box_x)
    l_y = random.randint(0, y_size - box_y)
    l_c = random.randint(0, c_size - 1)

    # Create a random matrix and apply probability threshold
    matrix = np.random.rand(box_x, box_y) < prob

    # Extract locations where the condition is True
    locs = np.argwhere(matrix)

    # Write to fi_locations.txt based on calculated faults
    with open("./fi_locations.txt", "w") as file:
        for loc in locs:
            fi_bit = 0 if random.random() <= 0.59 else random.randint(1, 7) # do +-1 with prob of 59% and other bitflips otherwise.
            file.write(f"{l_c} {loc[0] + l_x} {loc[1] + l_y} {fi_bit}\n")

    # Update mode file with layer and mode
    with open("./fi_mode.txt", "w") as file:
        file.write(f"injection {layer}\n")
    
    return layer_name, 0,  x_size * y_size, num_ops