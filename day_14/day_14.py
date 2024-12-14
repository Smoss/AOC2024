import operator
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import log10
from PIL import Image

import numpy as np

digits = list(map(str, range(0, 10)))

mult_log = log10(2024)
x, y = 101, 103

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
class Vector:
    row: int
    col: int

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(
            self.row + other.row,
            self.col + other.col
        )

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(
            self.row - other.row,
            self.col - other.col
        )

    def __mul__(self, other: int) -> "Vector":
        return Vector(
            self.row * other,
            self.col * other,
        )

@dataclass(frozen=True, eq=True)
class Bot:
    position: Vector
    velocity: Vector


    def advance(self) -> "Bot":
        new_position = self.position + self.velocity
        new_position_x, new_position_y = new_position.col, new_position.row
        new_position_x = new_position_x % x
        new_position_y = new_position_y % y
        return Bot(
            Vector(new_position_y, new_position_x),
            self.velocity
        )

cardinals = [
    Vector(1, 0),
    Vector(0, 1),
    Vector(-1, 0),
    Vector(0, -1)
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

def check_antinodes(nodes: set[Vector]) -> set[Vector]:
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


def check_antinodes_with_harmonic(nodes: set[Vector]) -> set[Vector]:
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
    bots = []
    for line in lines:
        pos, vel = line.split(' ')
        pos_x, pos_y = re.findall('-?\d+', pos)
        vel_x, vel_y = re.findall('-?\d+', vel)
        bots.append(
            Bot(
                Vector(int(pos_y), int(pos_x)),
                Vector(int(vel_y), int(vel_x)),
            )
        )
    for _ in range(100):
        bots = [
            bot.advance() for bot in bots
        ]
    quad_1, quad_2, quad_3, quad_4 = 0, 0, 0, 0
    for bot in bots:
        if bot.position.col < x // 2 and bot.position.row < y//2:
            quad_1 += 1
        if bot.position.col > x // 2 and bot.position.row < y//2:
            quad_2 += 1
        if bot.position.col < x // 2 and bot.position.row > y//2:
            quad_3 += 1
        if bot.position.col > x // 2 and bot.position.row > y//2:
            quad_4 += 1
    tot = quad_1 * quad_2 * quad_3 * quad_4
    return tot

def part_2(lines: list[str]) -> int:
    tot = 0
    bots = []
    for line in lines:
        pos, vel = line.split(' ')
        pos_x, pos_y = re.findall('-?\d+', pos)
        vel_x, vel_y = re.findall('-?\d+', vel)
        bots.append(
            Bot(
                Vector(int(pos_y), int(pos_x)),
                Vector(int(vel_y), int(vel_x)),
            )
        )
    for idx in range(100000):
        bots = [
            bot.advance() for bot in bots
        ]
        field = [[0 for _ in range(x)] for _ in range(y)]
        for bot in bots:
            field[bot.position.row][bot.position.col] = 255
        field_image = np.array(field, np.uint8)
        with Image.fromarray(field_image) as new_img:
            # new_img.
            new_img.save(
                f"output/{idx}.png"
            )
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")