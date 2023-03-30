# Quick and dirty script to randomly generate dataset splits

import os
import random
from math import ceil


if __name__ == "__main__":
    base = "../data/ShapeNet"

    splits = {
        "train": 83,
        "validation": 12,
        "test": 5
    }

    data = {}
    dirs = [os.path.basename(os.path.normpath(x[0])) for x in os.walk(base)]
    random.shuffle(dirs)
    last = 0

    total_weight = sum(splits.values())

    for split, weight in splits.items():
        entries = ceil(len(dirs) * (weight / total_weight))

        path = os.path.normpath(f"{base}/{split}.lst");
        with open(path, 'w') as file:
            for fn in dirs[last : last + entries]:
                    file.write(f"{fn}\n")

        last += entries
