from typing import List, Union

from constrained_attacks.datasets.core import Dataset
from constrained_attacks.datasets.ctu_13_neris import Ctu13Neris
from constrained_attacks.datasets.lcld_v2_dataset import LcldV2TimeDataset
from constrained_attacks.datasets.malware_dataset import MalwareDataset
from constrained_attacks.datasets.url_dataset import UrlDataset

datasets = {
    "lcld_v2_time": LcldV2TimeDataset(),
    "ctu_13_neris": Ctu13Neris(),
    "url": UrlDataset(),
    "malware": MalwareDataset(),
}


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name in datasets:
        return datasets[dataset_name]
    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' is not available."
        )


def load_datasets(dataset_names: Union[str, List[str]]):

    if isinstance(dataset_names, str):
        return load_dataset(dataset_names)
    else:
        return [
            {"name": dataset_name, "dataset": load_dataset(dataset_name)}
            for dataset_name in dataset_names
        ]
