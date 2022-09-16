import abc
import os
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder

from constrained_attacks.constraints.constraints import Constraints
from constrained_attacks.datasets.processing import IdentityTransformer
from constrained_attacks.utils.autoload import autoload

MLC_DATA_PATH = "./data/mlc/"


def check_folder_ready():
    mlc_file_path = f"{MLC_DATA_PATH}/.mlc"
    if os.path.isdir(MLC_DATA_PATH):
        if not os.path.isfile(mlc_file_path):
            raise Exception(
                "Directory is not empty and was not initialize by mlc."
            )
    else:
        Path(MLC_DATA_PATH).mkdir(parents=True, exist_ok=True)
        Path(mlc_file_path).touch()
    gitignore_path = f"{MLC_DATA_PATH}/.gitignore"
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, "w") as f:
            f.write("*")


class Dataset(ABC):
    def __init__(self, targets, date=None):
        self.targets = targets
        self.date = date
        self.data = None
        self.date_format = None
        self.drop_date = True

    @abc.abstractmethod
    def _load_data(self):
        pass

    def _get_splits(self):
        raise NotImplementedError

    def get_preprocessor(self):
        return IdentityTransformer()

    def get_data(self):
        self._load_data()
        return self.data.copy()

    def get_metadata(self):
        return None

    def get_x_y(self):
        data = self.get_data()
        y = data[self.targets].copy()
        for col in y.columns:
            y[col] = LabelEncoder().fit_transform(y[col].ravel())
        y = y.to_numpy()
        if len(self.targets) == 1:
            y = y.ravel()

        col_drop_x = self.targets.copy()
        if (self.date is not None) and self.drop_date:
            col_drop_x += [self.date]
        x = data.drop(columns=col_drop_x).copy()
        return x, y

    def get_x_y_t(self):
        x, y = self.get_x_y()
        if self.date is None:
            t = pd.DataFrame(
                data=np.arange(len(self.data)).reshape(-1, 1), columns=["date"]
            )["date"]

        else:
            t = pd.to_datetime(self.data[self.date], format=self.date_format)
        return x, y, t

    def get_splits(self):
        return self._get_splits()

    def get_constraints(self) -> Constraints:
        return None


class FileDataset(Dataset, ABC):
    def __init__(
        self,
        targets,
        data_path,
        metadata_path=None,
        date=None,
        date_format=None,
    ):
        super().__init__(targets, date)
        self.date_format = date_format
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.data = None
        self.metadata = None

    def _load_data(self):
        if self.data is None:
            self.data = autoload(self.data_path)

    def get_metadata(self):
        if self.metadata is None:
            self.metadata = pd.read_csv(self.metadata_path)
        return self.metadata


class DownloadFileDataset(FileDataset, ABC):
    def __init__(
        self,
        targets,
        data_path,
        data_url,
        metadata_path=None,
        metadata_url=None,
        date=None,
        date_format=None,
    ):
        self.data_url = data_url
        self.metadata_url = metadata_url

        super().__init__(targets, data_path, metadata_path, date, date_format)

    def _load_data(self):
        check_folder_ready()
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        if self.data is None and (not os.path.exists(self.data_path)):
            data = requests.get(self.data_url)
            with open(self.data_path, "wb") as file:
                file.write(data.content)
        super(DownloadFileDataset, self)._load_data()

    def get_metadata(self):
        check_folder_ready()
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        if (
            (not os.path.exists(self.metadata_path))
            and (self.metadata_url is not None)
            and (self.metadata_path is not None)
        ):
            data = requests.get(self.metadata_url)
            with open(self.metadata_path, "wb") as file:
                file.write(data.content)
        return super(DownloadFileDataset, self).get_metadata()
