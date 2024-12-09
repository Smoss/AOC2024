import operator
from collections import defaultdict
from dataclasses import dataclass

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
    max_i = len(lines)
    max_j = len(lines[0])
    nodes_dict = defaultdict(set)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char != '.':
                nodes_dict[char].add(Position(i, j))
    antinodes = set()
    for _, nodes in nodes_dict.items():
        antinodes = antinodes.union(check_antinodes(nodes))
    antinodes = {
        antinode for antinode in antinodes if
        0 <= antinode.row < max_i and 0 <= antinode.col < max_j
    }
    raw_lines = [
        [char for char in line] for line in lines
    ]
    for antinode in antinodes:
        raw_lines[antinode.row][antinode.col] = '#'
    print(
        '\n'.join(
            [
            ''.join(line) for line in raw_lines
            ]
        )
    )
    return len(antinodes)

def part_2(lines: list[str]) -> int:
    tot = 0
    max_i = len(lines)
    max_j = len(lines[0])
    nodes_dict = defaultdict(set)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char != '.':
                nodes_dict[char].add(Position(i, j))
    antinodes = set()
    for _, nodes in nodes_dict.items():
        antinodes = antinodes.union(check_antinodes_with_harmonic(nodes))
    antinodes = {
        antinode for antinode in antinodes if
        0 <= antinode.row < max_i and 0 <= antinode.col < max_j
    }
    raw_lines = [
        [char for char in line] for line in lines
    ]
    for antinode in antinodes:
        raw_lines[antinode.row][antinode.col] = '#'
    print(
        '\n'.join(
            [
            ''.join(line) for line in raw_lines
            ]
        )
    )
    return len(antinodes)



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")