import numpy as np
import pandas as pd

list_1, list_2 = [], []
with open("input", "r") as input_list:
    for line in input_list.readlines():
        val_1, val_2 = line.split("   ")
        list_1.append(int(val_1))
        list_2.append(int(val_2))
np_array_1 = pd.Series(np.sort(list_1))
np_array_2 = pd.Series(np.sort(list_2))

def part_1():
    return  np.sum(np.absolute(np_array_1 - np_array_2))

def part_2():
    vals_2 = np_array_1.apply(
        lambda val: val * len(np_array_2[np_array_2 == val])
    )
    return (np.sum(vals_2))

print(part_1())
print(part_2())