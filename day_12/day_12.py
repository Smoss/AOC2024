import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

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
    Position(0, -1),
]
up_down = [
    Position(1, 0),
    Position(-1, 0),
]
left_right = [
    Position(0, 1),
    Position(0, -1),
]
vert_side = {
    "D": Position(1, 0),
    "U": Position(-1, 0),
}
hor_side = {
    "R": Position(0, 1),
    "L": Position(0, -1),
}

new_neighbor_count = {
    1: 2,
    2: 0,
    3: -2,
    4: -4
}
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

def get_neigbhors(loc: Position, directions: Optional[list[Position]] = None) -> set[Position]:
    directions = directions or cardinals
    return {
        loc + direction for direction in directions
    }

# def get_neig

def part_1(lines: list[str]) -> int:
    tot = 0
    crop_spots = defaultdict(set)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            crop_spots[char].add(Position(i, j))
    regions = set()
    for crop, spots in crop_spots.items():
        while spots:
            start = next(iter(spots))
            curr_locs = {start}
            region = {start}
            perimeter = 4
            while curr_locs:
                new_locs = set()
                for curr_loc in curr_locs:
                    for neighbor in get_neigbhors(curr_loc):
                        if neighbor not in region and neighbor in spots:
                            new_locs.add(neighbor)
                            region.add(neighbor)
                            # perimeter += 3
                            new_neighbors = set()
                            for new_neighbor in get_neigbhors(neighbor):
                                if new_neighbor in region:
                                    new_neighbors.add(new_neighbor)
                            perimeter += new_neighbor_count[len(new_neighbors)]
                curr_locs = new_locs
            regions.add(frozenset(region))
            tot += perimeter * len(region)
            spots = spots - region
    return tot

def part_2(lines: list[str]) -> int:
    tot = 0
    crop_spots = defaultdict(set)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            crop_spots[char].add(Position(i, j))
    regions = set()
    for crop, spots in crop_spots.items():
        while spots:
            start = next(iter(spots))
            curr_locs = {start}
            region = {start}
            while curr_locs:
                new_locs = set()
                for curr_loc in curr_locs:
                    for neighbor in get_neigbhors(curr_loc):
                        if neighbor not in region and neighbor in spots:
                            new_locs.add(neighbor)
                            region.add(neighbor)
                curr_locs = new_locs
            regions.add(frozenset(region))
            spots = spots - region
    for region in regions:
        visited = set()
        sides = set()
        for loc in region:
            neighbors = get_neigbhors(loc)
            # if loc in visited:
            #     continue
            # visited.add(loc)
            if len(neighbors.intersection(region)) < 4:
                for key, check_direction in hor_side.items():
                    up_down_side = set()
                    for direction in up_down:
                        pointer = loc
                        while len((check_neigbors := get_neigbhors(pointer, [check_direction])).intersection(region)) < 1 \
                                        and pointer in region:
                            up_down_side.add(pointer)
                            pointer = pointer + direction
                    if up_down_side:
                        sides.add((key, frozenset(up_down_side)))
                for key, check_direction in vert_side.items():
                    left_right_side = set()
                    for direction in left_right:
                        pointer = loc
                        while len(get_neigbhors(pointer, [check_direction]).intersection(region)) < 1 \
                                        and pointer in region:
                            left_right_side.add(pointer)
                            pointer = pointer + direction
                    if left_right_side:
                        sides.add((key, frozenset(left_right_side)))
        tot += len(sides) * len(region)

    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")