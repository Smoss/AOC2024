import operator
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

operators = [
    operator.add,
    operator.mul,
]

def check(stack: list[str], targ: int, curr_val: int) -> int:
    if stack:
        next_val = stack[0]
        add = check(
            stack[1:],
            targ,
            curr_val + int(next_val)
        )
        mul = check(
            stack[1:],
            targ,
            curr_val * int(next_val)
        )
        return add + mul
    elif targ ==curr_val:
        return 1
    return 0


def check_with_cat(stack: list[str], targ: int, curr_val: int) -> int:
    if curr_val > targ:
        return 0
    if stack:
        next_val = stack[0]
        add = check_with_cat(
            stack[1:],
            targ,
            curr_val + int(next_val)
        )
        mul = check_with_cat(
            stack[1:],
            targ,
            curr_val * int(next_val)
        )
        cat = check_with_cat(
            stack[1:],
            targ,
            int(str(curr_val) + next_val)
        )
        return add + mul + cat
    elif targ ==curr_val:
        return 1
    return 0

def part_1(lines: list[str]) -> int:
    tot = 0
    for line in lines:
        targ_raw, stack_raw = line.split(': ')
        stack = stack_raw.split(' ')
        targ = int(targ_raw)
        tot += targ if check(stack, targ, 0) > 0 else 0
    return tot

def part_2(lines: list[str]) -> int:
    tot = 0
    for line in lines:
        print(line)
        targ_raw, stack_raw = line.split(': ')
        stack = stack_raw.split(' ')
        targ = int(targ_raw)
        tot += targ if check_with_cat(stack, targ, 0) > 0 else 0
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")