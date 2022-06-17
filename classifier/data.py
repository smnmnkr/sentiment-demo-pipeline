import logging
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


@dataclass
class Data(Dataset):
    data_path: str
    polarities: dict

    data_label: str
    target_label: str

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):
        logging.info(f'> Load/Init from {self.data_path}')
        self.data: pd.DataFrame = pd.read_csv(self.data_path)

    #  -------- __getitem__ -----------
    #
    def __getitem__(self, idx) -> T_co:
        return (
            self.data.iloc[[idx]],
            self.data[self.data_label][idx],
            self.data[self.target_label][idx]
        )

    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.data)

    #  -------- encode_label -----------
    #
    def encode_label(self, label: str) -> int:
        return self.polarities.get(label)

    #  -------- decode_label -----------
    #
    def decode_label(self, label: int) -> str:
        return {v: k for k, v in self.polarities.items()}.get(label)

    #  -------- get_label_keys -----------
    #
    def get_label_keys(self) -> set:
        return set(k for k in self.polarities.keys())

    #  -------- get_label_values -----------
    #
    def get_label_values(self) -> set:
        return set(k for k in self.polarities.values())
