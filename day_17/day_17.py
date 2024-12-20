import operator
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import log10
from typing import Any

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

@dataclass(frozen=True, eq=True)
class MotionVector:
    position: Vector
    direction_idx: int = 1

    def move(self) -> "MotionVector":
        return MotionVector(
            self.position + dirs[self.direction_idx],
            self.direction_idx,
        )


    def rotate_left(self) -> "MotionVector":
        return MotionVector(
            self.position,
            (self.direction_idx + 1) % 4,
        )

    def rotate_right(self) -> "MotionVector":
        return MotionVector(
            self.position,
            (self.direction_idx - 1) % 4,
        )

    def __add__(self, other: "Vector") -> "MotionVector":
        return MotionVector(
            self.position + other,
            self.direction_idx
        )

    def __sub__(self, other: "Vector") -> "MotionVector":
        return MotionVector(
            self.position - other,
            self.direction_idx
        )

    def __mul__(self, other: int) -> "MotionVector":
        return MotionVector(
            self.position * other,
            self.direction_idx
        )

@dataclass(frozen=True, eq=True)
class Reindeer:
    motion_vector: MotionVector
    path: frozenset[Vector]
    cost: int = 0

    def move(self) -> "Reindeer":
        return Reindeer(
            self.motion_vector + dirs[self.motion_vector.direction_idx],
            self.path.union(frozenset([self.motion_vector.position])),
            self.cost + 1
        )

    def rotate_left(self) -> "Reindeer":
        return Reindeer(
            self.motion_vector.rotate_left(),
            self.path.union(frozenset([self.motion_vector.position])),
            self.cost + 1000
        )

    def rotate_right(self) -> "Reindeer":
        return Reindeer(
            self.motion_vector.rotate_right(),
            self.path.union(frozenset([self.motion_vector.position])),
            self.cost + 1000
        )


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

@dataclass
class Computer:
    ops: list[int]
    output: list[int]
    reg_A: int
    reg_B: int
    reg_C: int
    instruction_idx: int = 0


    def get_combo_op(self) -> int:
        combo_op = self.ops[self.instruction_idx + 1]
        if combo_op == 4:
            return self.reg_A
        elif combo_op == 5:
            return self.reg_B
        elif combo_op == 6:
            return self.reg_C
        return combo_op

    def adv(self) -> None:
        denom = self.get_combo_op()
        self.reg_A //= (2 ** denom)
        self.instruction_idx += 2

    def bxl(self) -> None:
        literal_op = self.ops[self.instruction_idx + 1]
        self.reg_B  = self.reg_B ^ literal_op
        self.instruction_idx += 2

    def bst(self) -> None:
        self.reg_B = self.get_combo_op() % 8
        self.instruction_idx += 2

    def jnz(self) -> None:
        if self.reg_A == 0:
            self.instruction_idx += 2
        else:
            self.instruction_idx = self.ops[self.instruction_idx + 1]

    def bxc(self) -> None:
        self.reg_B = self.reg_B ^ self.reg_C
        self.instruction_idx += 2

    def out(self) -> None:
        self.output.append(self.get_combo_op() % 8)
        self.instruction_idx += 2

    def bdv(self) -> None:
        denom = self.get_combo_op()
        self.reg_B = self.reg_A // (2 ** denom)
        self.instruction_idx += 2

    def cdv(self) -> None:
        denom = self.get_combo_op()
        self.reg_C = self.reg_A // (2 ** denom)
        self.instruction_idx += 2

    def run_until_terminate(self) -> None:
        while self.instruction_idx < len(self.ops) - 1:
            op_code = self.ops[self.instruction_idx]
            if op_code == 0:
                self.adv()
            elif op_code == 1:
                self.bxl()
            elif op_code == 2:
                self.bst()
            elif op_code == 3:
                self.jnz()
            elif op_code == 4:
                self.bxc()
            elif op_code == 5:
                self.out()
            elif op_code == 6:
                self.bdv()
            elif op_code == 7:
                self.cdv()


    def run_until_terminate_with_check(self) -> None:
        while self.instruction_idx < len(self.ops) - 1:
            if self.output:
                chk_size = len(self.output)
                if self.ops[:chk_size] != self.output:
                    return
            op_code = self.ops[self.instruction_idx]
            if op_code == 0:
                self.adv()
            elif op_code == 1:
                self.bxl()
            elif op_code == 2:
                self.bst()
            elif op_code == 3:
                self.jnz()
            elif op_code == 4:
                self.bxc()
            elif op_code == 5:
                self.out()
            elif op_code == 6:
                self.bdv()
            elif op_code == 7:
                self.cdv()


def part_1(data: str) -> Any:
    tot = 0
    regs_raw, program_raw = data.split("\n\n")
    a, b, c = re.findall("\d+", regs_raw)
    ops = list(map(int, program_raw.replace("Program: ", "").split(',')))
    computer = Computer(
        ops,
        [],
        int(a),
        int(b),
        int(c),
    )
    computer.run_until_terminate()
    tot = ",".join(map(str, computer.output))
    return tot

def part_2(data: str) -> Any:
    tot = 0
    regs_raw, program_raw = data.split("\n\n")
    a, b, c = re.findall("\d+", regs_raw)
    ops = list(map(int, program_raw.replace("Program: ", "").split(',')))
    min_val = 1
    while True:
        computer = Computer(
            ops,
            [],
            min_val,
            int(b),
            int(c),
        )
        computer.run_until_terminate()
        if len(computer.output) == len(computer.ops):
            break
        min_val *= 8
    # while min_val
    print(min_val)
    for i in range(min_val, min_val * 8):
        computer = Computer(
            ops,
            [],
            i,
            int(b),
            int(c),
        )
        computer.run_until_terminate_with_check()
        chk_size = len(computer.output)
        if chk_size > 5:
            print(chk_size)
            print(computer.output)
            print(ops)
            print(i)
        # if chk_size > 10:
        #     print(computer.output)
        if computer.output == computer.ops:
            tot = i
            break
        if i % 10000 == 0:
            print(i)
        if i == 117440:
            print(computer.ops)
            print(computer.output)
            print(computer.ops == computer.output)
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")