import numpy as np
from numpy import ndarray


def load_data(filepath="input", delimiter=" ") -> list[ndarray]:
    try:
        lists = []
        with open(filepath) as input_file:
            for line in input_file.readlines():
                lists.append(np.array(list(map(int, line.split(" ")))))
        return lists
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Input file '{filepath}' not found") from err
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}") from e


def check_line(line: ndarray) -> bool:
    diffed = np.ediff1d(line)
    diffed_abs = np.absolute(diffed)
    return bool((np.all(diffed > 0) or np.all(diffed < 0)) and \
            np.all((diffed_abs <= 3) & (diffed_abs >= 1)))


def part_1(matrix: list[ndarray]) -> int:
    safes = 0
    for line in matrix:
        if check_line(line):
            safes += 1
    return safes

def part_2(matrix: list[ndarray]) -> int:
    safes = 0
    for line in matrix:
        if check_line(line):
            safes += 1
        else:
            for i in range(len(line)):
                if check_line(np.concatenate([line[:i], line[i+1:]])):
                    safes+=1
                    break
    return safes



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")