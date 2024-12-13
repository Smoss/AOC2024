import operator
import re
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

def load_data(filepath="input") -> str:
    try:
        with open(filepath) as input_file:
            return input_file.read()
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

def recur(
        curr_loc: Position,
        curr_tokens: int,
        prize: Position,
        buttons: list[Position],
        seen_positions: set[Position]
) -> int:
    if curr_loc == prize:
        return curr_tokens
    if curr_loc.row > prize.row or curr_loc.col > prize.col or curr_loc in seen_positions:
        return 0
    seen_positions.add(curr_loc)
    if (a_option := recur(
        curr_loc + buttons[0],
        curr_tokens + 3,
        prize,
        buttons,
        seen_positions
    )):
        return a_option

    if (
    b_option := recur(
        curr_loc + buttons[1],
        curr_tokens + 1,
        prize,
        buttons,
        seen_positions
    )):
        return b_option
    return 0

def part_1(input_data: str) -> int:
    tot = 0
    for game in input_data.split('\n\n'):
        *buttons_raw, prize_line = game.split('\n')
        buttons = []
        prize_x, prize_y = re.findall("\d+", prize_line)
        prize = Position(
            int(prize_y),
            int(prize_x)
        )
        for line in buttons_raw:
            button_x, button_y = re.findall("\d+", line)
            buttons.append(
                Position(
                    int(button_y),
                    int(button_x),
                )
            )
        button_a, button_b = buttons
        equations = np.array([
            [button_a.row, button_b.row],
            [button_a.col, button_b.col]
        ])
        values = np.array([
            prize.row, prize.col
        ])
        raw_result = np.linalg.solve(equations, values).astype(np.uint64)
        result = 0
        a_presses, b_presses = map(round, raw_result.tolist())
        # if
        if Position((a_presses * button_a.row + b_presses * button_b.row), (a_presses * button_a.col + b_presses * button_b.col)) == prize:
            result = np.dot(raw_result, [3, 1])
        tot += result
    return int(tot)


def part_2(input_data: str) -> int:
    tot = 0
    for game in input_data.split('\n\n'):
        *buttons_raw, prize_line = game.split('\n')
        buttons = []
        prize_x, prize_y = re.findall("\d+", prize_line)
        prize = Position(
            int(prize_y) + 10000000000000,
            int(prize_x) + 10000000000000
        )
        for line in buttons_raw:
            button_x, button_y = re.findall("\d+", line)
            buttons.append(
                Position(
                    int(button_y),
                    int(button_x),
                )
            )
        button_a, button_b = buttons
        equations = np.array([
            [button_a.row, button_b.row],
            [button_a.col, button_b.col]
        ], dtype=np.float64)
        values = np.array([
            prize.row, prize.col
        ], dtype=np.float64)
        raw_result = np.linalg.solve(equations, values)
        result = 0
        a_presses, b_presses = map(round, raw_result.tolist())
        if Position((a_presses * button_a.row + b_presses * button_b.row), (a_presses * button_a.col + b_presses * button_b.col)) == prize:
            result = np.dot(raw_result, [3, 1])
        else:
            print(raw_result)
            print(a_presses, b_presses)
            print(Position((a_presses * button_a.row + b_presses * button_b.row), (a_presses * button_a.col + b_presses * button_b.col)), prize)
        tot += result
    return int(tot)



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")