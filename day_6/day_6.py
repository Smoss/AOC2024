from dataclasses import dataclass

digits = list(map(str, range(0, 10)))

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


@dataclass
class Direction:
    vert: int
    hori: int

@dataclass(frozen=True, eq=True)
class Position:
    row: int
    col: int

    def __add__(self, other: Direction) -> "Position":
        return Position(
            self.row + other.vert,
            self.col + other.hori
        )

start_dirs = {
    "^": 0,
    ">": 1,
    "v": 2,
    "<": 3
}

dirs = [
    Direction(-1, 0),
    Direction(0, 1),
    Direction(1, 0),
    Direction(0, -1),
]

@dataclass(frozen=True, eq=True)
class Guard:
    pos: Position
    direction_idx: int

    def move(self) -> "Guard":
        return Guard(
            self.pos + dirs[self.direction_idx],
            self.direction_idx
        )

    def rotate(self) -> "Guard":
        return Guard(
            self.pos,
            (self.direction_idx + 1) % 4
        )

def set_stage(starting_state: list[str])-> tuple[Guard, set[Position]]:
    blocks = set()
    guard = Guard(Position(0,0), 0)
    for i, line in enumerate(starting_state):
        for j, char in enumerate(line):
            if char == "#":
                blocks.add(Position(i, j))
            elif char in start_dirs:
                guard = Guard(Position(i, j), start_dirs[char])
    return guard, blocks

def run_sim(guard: Guard, blocks: set[Position], max_row: int, max_col: int) -> tuple[Guard, set[Position], bool]:
    locs = set()
    guards = set()
    looped = False
    while guard not in guards and 0 <= guard.pos.row < max_row and 0 <= guard.pos.col < max_col:
        guards.add(guard)
        locs.add(guard.pos)
        new_guard = guard.move()
        if new_guard.pos in blocks:
            new_guard = guard.rotate().move()
        if new_guard.pos in blocks:
            new_guard = guard.rotate().rotate().move()
        if new_guard.pos in blocks:
            new_guard = guard.rotate().rotate().rotate().move()
        guard = new_guard
        if guard in guards:
            looped = True
        if guard.pos in blocks:
            raise AssertionError("Guard in block")

    return guard, locs, looped


def part_1(starting_state: list[str]) -> int:
    guard, blocks = set_stage(starting_state)
    max_row = len(starting_state)
    max_col = len(starting_state[0])
    _, locs, _ = run_sim(guard, blocks, max_row, max_col)

    block_list = [
        list(line) for line in starting_state
    ]
    for block in locs:
        block_list[block.row][block.col] = 'X'
    print(
        "\n".join([
            "".join(line) for line in block_list
        ])
    )
    # assert guard.pos not in locs
    assert len(blocks.intersection(locs)) == 0, print(blocks.intersection(locs))
    return len(locs)

def part_2(starting_state: list[str]) -> int:
    print('*****')
    guard, blocks = set_stage(starting_state)
    max_row = len(starting_state)
    max_col = len(starting_state[0])
    potential_blocks = set()
    for i in range(max_row):
        for j in range(max_col):
            potential_blocks.add(Position(i, j))
    last_guard, potential_blocks, _ = run_sim(
        guard,
        blocks,
        max_row,
        max_col
    )
    potential_blocks.remove(guard.pos)
    potential_blocks.add(last_guard.pos)
    blockers = set()
    total_checks = len(potential_blocks)
    print(total_checks)
    for idx, block in enumerate(potential_blocks):
        if idx % 100 == 0:
            print(f"{idx / total_checks:.2f}")
        _, locs, looped = run_sim(
            guard,
            blocks.union({block}),
            max_row,
            max_col
        )
        assert block not in locs
        assert len(blocks.intersection(locs)) == 0, print(blocks.intersection(locs))
        if looped:
            blockers.add(block)


    block_list = [
        list(line) for line in starting_state
    ]
    for block in blockers:
        block_list[block.row][block.col] = 'O'
    print(
        "\n".join([
            "".join(line) for line in block_list
        ])
    )
    return len(blockers)



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")