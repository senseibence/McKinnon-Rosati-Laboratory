import argparse
from pathlib import Path
import numpy as np

directory = Path("../Arrays")
def print_trainable(dataset):
    path = f"{dataset}_train_labels_hvg_*.npy"
    for array in directory.glob(path):
        y = np.load(array)
        if (len(np.unique(y)) < 2): continue
        name = array.stem.removeprefix(f"{dataset}_train_labels_hvg_")
        print(dataset + " " + name)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
args = ap.parse_args()
print_trainable(args.dataset)