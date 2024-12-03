import re

digits = list(map(str, range(0, 10)))

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


def part_1(ops: list[str]) -> int:
    tot = 0
    for op_line in ops:
        results = re.findall("mul\(\d\d?\d?,\d\d?\d?\)", op_line)
        for op in results:
            left, right = op[4:-1].split(',')
            print(op)
            tot += int(left) * int(right)
    return tot

def part_2(ops: list[str]) -> int:
    tot = 0
    act = True
    for op_line in ops:
        i = 0
        while i < len(op_line):
            left = 0
            right = 0
            if op_line[i:i+4] == "do()":
                act = True
            if op_line[i:i+7] == "don't()":
                act = False
            if op_line[i:i+4]=="mul(" and act:
                inc = 4
                while op_line[i+inc] in digits and inc < 9:
                    left = left * 10 + int(op_line[i+inc])
                    inc += 1
                if op_line[i+inc] != ',':
                    i+= inc
                    continue
                right_inc = inc + 1
                while op_line[i+right_inc] in digits and right_inc < inc + 4:
                    right = right * 10 + int(op_line[i+right_inc])
                    right_inc += 1
                print(op_line[i: i + right_inc])
                if op_line[i+right_inc] != ')':
                    i+= right_inc
                    continue
                tot += right * left
                print(right * left)
                i += right_inc
            else:
                i += 1
    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(input_data)}")
    print(f"Part 2 result: {part_2(input_data)}")