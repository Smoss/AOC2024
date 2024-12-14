import operator
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

digits = list(map(str, range(0, 10)))

@dataclass
class Direction:
    vert: int
    hori: int

@dataclass(frozen=True, eq=True)
class Position:
    row: int
    col: int

    def __add__(self, other: "Position") -> "Position":
        return Position(
            self.row + other.row,
            self.col + other.col
        )

    def __sub__(self, other: "Position") -> "Position":
        return Position(
            self.row - other.row,
            self.col - other.col
        )

    def __mul__(self, other: int) -> "Position":
        return Position(
            self.row * other,
            self.col * other,
        )

@dataclass
class SimpleFile:
    length: int
    index: int
    file_id: int

    @property
    def checksum_val(self) -> int:
        return sum([self.file_id * i for i in range(self.index, self.index + self.length)])

def load_data(filepath="input") -> list[str]:
    try:
        with open(filepath) as input_file:
            return [
                line.strip() for line in input_file.readlines()
            ]
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Input file '{filepath}' not found") from err
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}") from e

def check_antinodes(nodes: set[Position]) -> set[Position]:
    antinodes = set()
    nodes_list = list(nodes)
    for i, node in enumerate(nodes_list):
        for second_node in nodes_list[i+1:]:
            dist = node - second_node
            antinode_1 = node + dist
            antinode_2 = second_node + (dist * -1)
            antinodes.add(antinode_1)
            antinodes.add(antinode_2)
    return antinodes


def check_antinodes_with_harmonic(nodes: set[Position]) -> set[Position]:
    antinodes = set()
    nodes_list = list(nodes)
    for i, node in enumerate(nodes_list):
        antinodes.add(node)
        for second_node in nodes_list[i+1:]:
            dist = node - second_node
            for j in range(1,100):
                nu_dist = (dist * j)
                antinodes.add(node + nu_dist)
            for j in range(1, 100):
                antinodes.add(second_node + (dist * -1 * j))
    return antinodes

def part_1(lines: list[str]) -> int:
    tot = 0
    total_len = 0
    line, *_ = lines
    files = []
    emptys = []
    file_system = np.zeros(0)
    for idx, char in enumerate(line):
        if idx % 2 ==0:
            files.append(int(char))
            file_system = np.concat([file_system, np.ones(int(char)) * (idx//2)])
        else:
            file_system = np.concat([file_system, np.zeros(int(char)) - 1])
            emptys.append(int(char))
    idx = files.pop(0)
    end_point = len(file_system) - 1
    editable_file_system = file_system.tolist()
    while end_point > idx:
        if editable_file_system[idx] >= 0:
            idx += files.pop(0)
        elif editable_file_system[end_point] == -1:
            end_point -= emptys.pop()
        else:
            editable_file_system[idx] = editable_file_system[end_point]
            editable_file_system[end_point] = -1
            idx +=1
            end_point -= 1

    editable_file_system = [idx * val for idx, val in enumerate(editable_file_system) if val >= 0]

    # print(editable_file_system)
    # print(sum(editable_file_system))
    return int(sum(editable_file_system))

def part_2(lines: list[str]) -> int:
    tot = 0
    line, *_ = lines
    files = []
    emptys = []
    # file_system = np.zeros(0)
    seek_idx = 0
    empty_zones = {}
    files = []
    file_system = np.zeros(0)
    for idx, char in enumerate(line):
        section_len = int(char)
        if idx % 2 ==0:
            file_id = idx//2
            files.append(
                SimpleFile(
                    section_len,
                    seek_idx,
                    file_id
                )
            )

            file_system = np.concat([file_system, np.zeros(int(char)) - 1])
        else:
            file_system = np.concat([file_system, np.zeros(int(char)) - 1])
            empty_zones[seek_idx] = section_len
            emptys.append(seek_idx)
        seek_idx += section_len
    files_to_check = files[1:]
    editable_file_system = file_system.tolist()
    while files_to_check:
        file_to_check = files_to_check.pop()
        file_len = file_to_check.length
        for idx, empty_idx in enumerate(emptys):
            empty_size = empty_zones[empty_idx]
            if empty_size >= file_len and empty_idx < file_to_check.index:
                del empty_zones[empty_idx]
                empty_zones[file_to_check.index] = file_to_check.length
                new_size = empty_size - file_len
                file_to_check.index = empty_idx
                if new_size > 0:
                    empty_zones[empty_idx + file_len] = new_size
                break
        emptys = [key for key in empty_zones]
        emptys.sort()
    # editable_file_system = [idx * val for idx, val in enumerate(editable_file_system) if val >= 0]

    # print(files)
    # print(emptys)
    # print(sum([file.checksum_val for file in files]))
    # for file in files:
    #     for i in range(file.index, file.index + file.length):
    #         editable_file_system[i] = file.file_id

    # print("".join(map(str, map(int, editable_file_system))).replace("-1", "."))
    return sum([file.checksum_val for file in files])



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")