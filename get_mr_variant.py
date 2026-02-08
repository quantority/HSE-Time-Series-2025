#! /usr/bin/env python3

import string
import sys

tasks = list(range(111, 115))
tasks.append(116)

if __name__ == '__main__':
    result = 0
    for symbol in sys.argv[1]:
        for num, elem in enumerate(string.ascii_lowercase):
            if symbol == elem:
                result += num
                break
    print(tasks[result % len(tasks)])
