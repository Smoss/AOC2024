import operator
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

digits = list(map(str, range(0, 10)))

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


cardinals = [
    Position(1, 0),
    Position(0, 1),
    Position(-1, 0),
    Position(0, -1)
]
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
    height_map = {}
    # visited = set()
    # reached_peaks = set()
    starting_positions = set()
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char != '.':
                height_map[Position(i, j)] = int(char)
            if char == '0':
                starting_positions.add(Position(i , j))
    # curr_len = 0
    for starting_position in starting_positions:
        visited = set()
        curr_locs = {starting_position}
        while curr_locs:
            # visited = visited.union(curr_locs)
            new_neighbors = set()
            for curr_loc in curr_locs:
                curr_height = height_map[curr_loc]
                if curr_height == 9:
                    tot += 1
                else:
                    for direction in cardinals:
                        new_loc = direction + curr_loc
                        if new_loc in height_map and height_map[new_loc] == curr_height + 1:
                            new_neighbors.add(new_loc)
            curr_locs = new_neighbors
    return tot

def part_2(lines: list[str]) -> int:
    tot = 0
    height_map = {}
    # visited = set()
    # reached_peaks = set()
    curr_locs = []
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char != '.':
                height_map[Position(i, j)] = int(char)
            if char == '0':
                curr_locs.append(Position(i , j))
    # curr_len = 0
    # for starting_position in starting_positions:
    # visited = set()
    # curr_locs = []
    while curr_locs:
        # visited = visited.union(curr_locs)
        new_neighbors = []
        for curr_loc in curr_locs:
            curr_height = height_map[curr_loc]
            if curr_height == 9:
                tot += 1
            else:
                for direction in cardinals:
                    new_loc = direction + curr_loc
                    if new_loc in height_map and height_map[new_loc] == curr_height + 1:
                        new_neighbors.append(new_loc)
        curr_locs = new_neighbors
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")