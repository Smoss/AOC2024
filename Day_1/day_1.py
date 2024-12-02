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

def part_2(array1: pd.Series, array2: pd.Series) -> float:
    """Calculate weighted sum where each value is multiplied by its occurrence count.
    
    Args:
        array1: First sorted array of numbers
        array2: Second sorted array of numbers
        
    Returns:
        Weighted sum based on occurrence counts
    """
    value_counts = array2.value_counts()
    return np.sum(array1 * array1.map(value_counts.get).fillna(0))

print(part_1())
print(part_2())