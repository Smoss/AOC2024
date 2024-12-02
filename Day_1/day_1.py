import numpy as np
import pandas as pd

def load_data(filepath="input", delimiter="   ") -> tuple[pd.Series, pd.Series]:
    try:
        data = pd.read_csv(filepath, header=None, sep=delimiter, 
                          names=['val_1', 'val_2'])
        return (pd.Series(np.sort(data['val_1'])), 
                pd.Series(np.sort(data['val_2'])))
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{filepath}' not found")
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}")


def part_1(array1: pd.Series, array2: pd.Series) -> float:
    return np.sum(np.absolute(array1 - array2)) # type: ignore

def part_2(array1: pd.Series, array2: pd.Series) -> float:
    """Calculate weighted sum where each value is multiplied by its occurrence count.
    
    Args:
        array1: First sorted array of numbers
        array2: Second sorted array of numbers
        
    Returns:
        Weighted sum based on occurrence counts
    """
    value_counts = array2.value_counts()
    return np.sum(array1 * array1.map(value_counts.get).fillna(0)) # type: ignore

input_data = load_data()
print(part_1(*input_data))
print(part_2(*input_data))