import pathlib
import pickle

import pandas as pd


def autoload(path):
    extension = pathlib.Path(path).suffix
    print(extension)
    if extension == ".csv":
        return pd.read_csv(path)
    if extension == ".pkl" or extension == ".pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise NotImplementedError
