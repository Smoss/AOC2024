import re
from collections import defaultdict

digits = list(map(str, range(0, 10)))

def load_data(filepath="input") -> tuple[list[str], list[str]]:
    try:
        with open(filepath) as input_file:
            rules, updates = input_file.read().split('\n\n')
        return rules.split('\n'), updates.split('\n')
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Input file '{filepath}' not found") from err
    except Exception as e:
        raise ValueError(f"Error processing input data: {str(e)}") from e


def part_1(rules: list[str], updates: list[str]) -> int:
    tot = 0
    rule_sets = defaultdict(set)
    for rule in rules:
        pre, post = rule.split('|')
        rule_sets[pre].add(post)
    print(rule_sets)
    for update in updates:
        # print('*****')
        pages = update.split(',')
        # print(pages)
        seen = {pages[0]}
        median = len(pages) // 2
        for page in pages[1:]:
            is_good = True
            for pre_check in rule_sets[page]:
                if pre_check in seen:
                    is_good = False
                    break
            if not is_good:
                break
            seen.add(page)
        if len(seen) == len(pages):
            tot+= int(pages[median])

    return tot

def reorder_pages(pages: list[str], rule_sets: dict[str, set[str]]) -> tuple[list[str], bool]:
    swapped = False
    page_locs = {pages[0]: 0}
    for idx, page in enumerate(pages[1:]):
        page_locs[page] = idx + 1
        for pre_check in rule_sets[page]:
            if pre_check in page_locs:
                prev_loc = page_locs[pre_check]
                page_locs[pre_check] = idx + 1
                page_locs[page] = prev_loc
                swapped = True
                print(page, pre_check)
                break

    pages_in_order = ["" for _ in range(len(pages))]
    for page, idx in page_locs.items():
        pages_in_order[idx] = page
    return pages_in_order , swapped


def part_2(rules: list[str], updates: list[str]) -> int:
    tot = 0
    rule_sets = defaultdict(set)
    for rule in rules:
        pre, post = rule.split('|')
        rule_sets[pre].add(post)
    print(rule_sets)
    for update in updates:
        # print('*****')
        pages = update.split(',')
        # print(pages)
        median = len(pages) // 2
        pages, swapped = reorder_pages(pages, rule_sets)
        was_swapped = swapped
        while swapped:
            pages, swapped = reorder_pages(pages, rule_sets)
        if was_swapped:
            # print(pages)
            tot+= int(pages[median])

    return tot



if __name__ == '__main__':
    input_data = load_data()
    print(f"Part 1 result: {part_1(*input_data)}")
    print(f"Part 2 result: {part_2(*input_data)}")