"""Shared paths and small helpers."""
import os
import pickle
import time

ROOT = r"C:\Users\7mudi\Downloads\amazon ml challenge"
DATA = os.path.join(ROOT, "student_resource", "dataset")
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "alt", "output")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SRC_BASE = 10 ** 10  # record code = src * SRC_BASE + numeric id
N_FOLDS = 5
HOLDOUT_FOLD = 4

_T0 = time.time()


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s]", *args, flush=True)


def wpath(name):
    return os.path.join(WORK, name)


def save(obj, name):
    with open(wpath(name), "wb") as f:
        pickle.dump(obj, f, protocol=5)


def load(name):
    with open(wpath(name), "rb") as f:
        return pickle.load(f)


def exists(name):
    return os.path.exists(wpath(name))


def fold_of(s1_num):
    """Deterministic 5-fold assignment by S1 entity (group = entity)."""
    import numpy as np
    x = np.asarray(s1_num, dtype=np.uint64)
    x = (x * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(33)
    return (x % np.uint64(N_FOLDS)).astype(np.int8)
