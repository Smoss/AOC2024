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

    def gps_pos(self) -> int:
        return 100 * self.row + self.col

@dataclass(frozen=True, eq=True)
class Box:
    positions: tuple[Vector, Vector]

    def __add__(self, other: "Vector") -> "Box":
        l, r = self.positions
        new_box = Box(
            (
                l + other,
                r + other
            )
        )
        new_box.check()
        return new_box

    def check(self) -> None:
        l, r = self.positions
        assert l.row == r.row and l.col == r.col - 1, f"Bad box with positions {self.positions}"

    def gps_pos(self) -> int:
        return self.positions[0].gps_pos()

start_dirs = {
    "^": Vector(-1, 0),
    ">": Vector(0, 1),
    "v": Vector(1, 0),
    "<": Vector(0, -1),
}

dirs = [
    Vector(-1, 0),
    Vector(0, 1),
    Vector(1, 0),
    Vector(0, -1),
]
@dataclass(frozen=True, eq=True)
class Bot:
    position: Vector
    directions: tuple[Vector]
    list_idx: int = 0

    def move(self, boxes: set[Vector], statics: set[Vector]) -> "Bot":
        direction = self.directions[0]
        new_position = self.position + direction
        if new_position in boxes:
            new_box_position = new_position
            while new_box_position in boxes and new_box_position not in statics:
                new_box_position = new_box_position + direction
            if new_box_position in statics:
                return Bot(
                    self.position,
                    self.directions[1:]
                )
            boxes.remove(new_position)
            boxes.add(new_box_position)
            return Bot(
                new_position,
                self.directions[1:]
            )


        if new_position in statics:
            return Bot(
                self.position,
                self.directions[1:]
            )
        return Bot(
                new_position,
                self.directions[1:]
            )

    def move_wide(self, boxes: set[Box], statics: set[Vector]) -> "Bot":
        direction = self.directions[0]
        new_position = self.position + direction
        box_positions = {}
        for box in boxes:
            l, r = box.positions
            box_positions[l] = box
            box_positions[r] = box
        if new_position in box_positions:
            curr_boxes = {box_positions[new_position]}
            moved_boxes = set()
            new_boxes = set()
            visited_boxes = set()
            while curr_boxes:
                next_boxes = set()
                for curr_box in curr_boxes:
                    moved_boxes.add(curr_box)
                    new_box = curr_box + direction
                    new_boxes.add(curr_box + direction)
                    visited_boxes.add(curr_box)
                    for position in new_box.positions:
                        if position in statics:
                            return Bot(
                                self.position,
                                self.directions[1:]
                            )
                        if position in box_positions:
                            next_boxes.add(box_positions[position])
                curr_boxes = next_boxes - visited_boxes
            for box in moved_boxes:
                boxes.remove(box)
            for box in new_boxes:
                boxes.add(box)
            return Bot(
                new_position,
                self.directions[1:]
            )


        if new_position in statics:
            return Bot(
                self.position,
                self.directions[1:]
            )
        return Bot(
                new_position,
                self.directions[1:]
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

def load_data(filepath="input") -> str:
    try:
        with open(filepath) as input_file:
            return input_file.read()
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

def part_1(data: str) -> int:
    tot = 0
    statics = set()
    boxes = set()
    robot = None
    init_state_raw, moves_raw = data.split('\n\n')
    moves : list[Vector] = [
        start_dirs[char] for char in moves_raw.replace('\n', '').strip()
    ]
    for i, line in enumerate(init_state_raw.split('\n')):
        for j, char in enumerate(line):
            curr_pos = Vector(i, j)
            if char == "#":
                statics.add(curr_pos)
            elif char == 'O':
                boxes.add(curr_pos)
            elif char == "@":
                robot = Bot(
                    Vector(i, j),
                    tuple(moves)
                )
    idx = 0
    while robot.directions:
        robot = robot.move(boxes, statics)
        # map= [[
        #     (128, 128, 128) for _ in line
        # ] for line in init_state_raw.split('\n')]
        # for static in statics:
        #     map[static.row][static.col] = (255, 255, 0)
        # for box in boxes:
        #     map[box.row][box.col] = (0, 255, 255)
        # map[robot.position.row][robot.position.col] = (0, 0, 0)
        # field_image = np.array(map, np.uint8)
        # with Image.fromarray(field_image) as new_img:
        #     # new_img.
        #     new_img.resize(((i+1)*10, (j+1) *10)).save(
        #         f"output/{idx}.png"
        #     )
        # idx += 1
    tot = sum([box.gps_pos() for box in boxes])
    return tot

def part_2(data: str) -> int:
    tot = 0
    statics = set()
    boxes = set()
    robot = None
    init_state_raw, moves_raw = data.split('\n\n')
    moves : list[Vector] = [
        start_dirs[char] for char in moves_raw.replace('\n', '').strip()
    ]
    for i, line in enumerate(init_state_raw.split('\n')):
        for j, char in enumerate(line):
            curr_pos = Vector(i, j * 2)
            curr_pos_2 = Vector(i, j * 2 + 1)
            if char == "#":
                statics.add(curr_pos)
                statics.add(curr_pos_2)
            elif char == 'O':
                new_box = Box((curr_pos, curr_pos_2))
                new_box.check()
                boxes.add(new_box)
            elif char == "@":
                robot = Bot(
                    curr_pos,
                    tuple(moves)
                )
    frames = []
    idx = 0
    map_size = (i + 1) * ((j+ 1) * 2 + 1) * 3
    raw_map = np.ones(map_size) * 128
    raw_map = np.resize(raw_map, ((i+1), ((j+1) * 2) , 3))
    map = raw_map.tolist()
    for static in statics:
        map[static.row][static.col] = (255, 255, 0)
    for box in boxes:
        l, r = box.positions
        map[l.row][l.col] = (0, 255, 255)
        map[r.row][r.col] = (0, 255, 255)
    map[robot.position.row][robot.position.col] = (0, 0, 0)
    # field_image = np.array(map, np.uint8)
    # with Image.fromarray(field_image) as new_img:
    #     # new_img.
    #     new_img.resize(((j+1) *4, (i+1)*2)).save(
    #         f"output/{idx}.png"
    #     )
    frames.append(map)
    while robot.directions:
        robot = robot.move_wide(boxes, statics)
        # map= [[
        #     (128, 128, 128) for _ in line
        # ] for line in init_state_raw.split('\n')]
        map_size = (i + 1) * ((j+ 1) * 2 + 1) * 3
        raw_map = np.ones(map_size) * 128
        raw_map = np.resize(raw_map, ((i+1), ((j+1) * 2) , 3))
        map = raw_map.tolist()
        for static in statics:
            map[static.row][static.col] = (255, 255, 0)
        for box in boxes:
            l, r = box.positions
            map[l.row][l.col] = (0, 255, 255)
            map[r.row][r.col] = (0, 255, 255)
        map[robot.position.row][robot.position.col] = (0, 0, 0)
        # field_image = np.array(map, np.uint8)
        # idx += 1
        # with Image.fromarray(field_image) as new_img:
        #     # new_img.
        #     new_img.resize(((j+1) *4, (i+1)*2)).save(
        #         f"output/{idx}.png"
        #     )
        frames.append(map)
    tot = sum([box.gps_pos() for box in boxes])
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")