
# digits = list(map(str, range(0, 10)))
dirs = [
    (1,0),
    (1,1),
    (0,1),
    (-1,1),
    (-1,0),
    (-1,-1),
    (0,-1),
    (1,-1)
]
dir_pairs = [
    ((-1,1), (1,-1)),
    ((-1,-1), (1,1))
]

XMAS = "XMAS"
MAS = "MAS"

def load_data(filepath="input") -> list[str]:
    try:
        ops = []
        with open(filepath) as input_file:
            for line in input_file.readlines():
                ops.append(line)
        return ops
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Input file '{filepath}' not found") from err
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}") from e

def check_line(
        pos: tuple[int, int],
        dir: tuple[int, int],
        max_i: int,
        max_j: int,
        xmas_dict: dict[str, set[tuple[int, int]]]
) -> int:
    char_locs = []
    curr_char = 0
    while 0 <= pos[0] <= max_i and 0 <= pos[1] <= max_j:
        if pos in xmas_dict[XMAS[curr_char]]:
            char_locs.append(pos)
            curr_char += 1
        else:
            break
        pos = (pos[0] + dir[0], pos[1] + dir[1])
        prev_char_present = False
        for k in range(curr_char):
            if pos in xmas_dict[XMAS[k]]:
                prev_char_present = True
                break
        if curr_char > 3 or prev_char_present:
            break

    return 1 if curr_char > 3 else 0

def part_1(ops: list[str]) -> int:
    tot = 0
    xmas_dict: dict[str, set[tuple[int, int]]] = {
        char: set() for char in XMAS
    }
    max_i, max_j = 0, 0
    for i, line in enumerate(ops):
        for j, char in enumerate(line):
            if char in xmas_dict:
                xmas_dict[char].add((i,j))
            max_j = max(j, max_j)
        max_i = max(i, max_i)
    # print(xmas_dict)
    for pos in xmas_dict[XMAS[0]]:
        for dir in dirs:
            tot += check_line(
                pos,
                dir,
                max_i,
                max_j,
                xmas_dict
            )
            # print(tot)
    return tot


def check_x(
        pos: tuple[int, int],
        mas_dict: dict[str, set[tuple[int, int]]]
) -> int:
    one_x, two_x = False, False

    (diag_1_1, diag_1_2), (diag_2_1, diag_2_2) = dir_pairs
    pos_1_1 = (pos[0] + diag_1_1[0], pos[1] + diag_1_1[1])
    pos_1_2 = (pos[0] + diag_1_2[0], pos[1] + diag_1_2[1])
    if (pos_1_2 in mas_dict['S'] and pos_1_1 in mas_dict['M']) or (pos_1_1 in mas_dict['S'] and pos_1_2 in mas_dict['M']):
        one_x = True

    pos_2_1 = (pos[0] + diag_2_1[0], pos[1] + diag_2_1[1])
    pos_2_2 = (pos[0] + diag_2_2[0], pos[1] + diag_2_2[1])
    if (pos_2_2 in mas_dict['S'] and pos_2_1 in mas_dict['M']) or (
            pos_2_1 in mas_dict['S'] and pos_2_2 in mas_dict['M']):
        two_x = True
    return 1 if one_x and two_x else 0

def part_2(ops: list[str]) -> int:
    tot = 0
    mas_dict: dict[str, set[tuple[int, int]]]  = {
        char: set() for char in MAS
    }
    max_i, max_j = 0, 0
    for i, line in enumerate(ops):
        for j, char in enumerate(line):
            if char in mas_dict:
                mas_dict[char].add((i,j))
            max_j = max(j, max_j)
        max_i = max(i, max_i)
    # print(xmas_dict)
    for pos in mas_dict[MAS[1]]:
        if 0 < pos[0] < max_i and 0 < pos[1] < max_j:
                tot += check_x(
                    pos,
                    mas_dict
                )
            # print(tot)
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")