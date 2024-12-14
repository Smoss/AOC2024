import operator
from collections import defaultdict
from dataclasses import dataclass
from math import log10

import numpy as np

digits = list(map(str, range(0, 10)))

mult_log = log10(2024)

@dataclass(frozen=True, eq=True)
class Stone:
    logarithm: float

    def blink(self) -> list["Stone"]:
        if self.logarithm == -1.0:
            return [Stone(0.0)]
        mod_value = int(self.logarithm) % 2
        if mod_value == 1:
            curr_value = str(round(10 ** self.logarithm))
            mid = int(self.logarithm // 2)
            left_value = int(curr_value[:mid+1])
            right_value = int(curr_value[mid+1:])
            return [
                Stone(
                    log10(left_value)
                ),
                Stone(
                    log10(right_value) if right_value > 0 else -1
                )
            ]
        return [
            Stone(
                self.logarithm + mult_log
            )
        ]

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
    line, *_ = lines
    stones = []
    for stone_raw in line.split(' '):
        value = int(stone_raw)
        log_val = -1
        if value > 0:
            log_val = log10(value)
        stones.append(
            Stone(log_val)
        )
    for _ in range(25):
        new_stones = []
        for stone in stones:
            new_stones.extend(stone.blink())
        stones = new_stones
    tot = len(stones)
    return tot

def part_2(lines: list[str]) -> int:
    tot = 0
    line, *_ = lines
    stones = defaultdict(int)
    for stone_raw in line.split(' '):
        value = int(stone_raw)
        log_val = -1
        if value > 0:
            log_val = log10(value)
        stones[
            Stone(log_val)
        ] += 1
    for _ in range(75):
        new_stones = defaultdict(int)
        for stone, count in stones.items():
            for result in stone.blink():
                new_stones[result] += count
        stones = new_stones
    tot = sum([stone_count for stone_count in stones.values()])
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")