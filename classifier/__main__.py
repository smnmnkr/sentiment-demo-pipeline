import argparse
import logging
import random

import torch

from classifier import Data, Encoder, Model, Trainer
from classifier.util import dict_merge, load_json

from classifier.util import get_device


class Main:

    #  -------- __init__ -----------
    #
    def __init__(self):
        # --- ---------------------------------
        # --- base setup
        self.config: dict = Main.__load_config()
        Main.__load_logger(f'{self.config["out_path"]}full.log')
        Main.__setup_pytorch(self.config["seed"], self.config["cuda"])

        # --- ---------------------------------
        # --- load data
        logging.info(f'\n[--- LOAD DATA -> ({list(k for k in self.config["data"]["paths"].keys())}) ---]')
        self.data: dict = self.load_data()

        logging.info('\n[--- LOAD COMPONENTS ---]')
        # encoding, model
        self.encoding = Encoder(self.config['model']['encoding'])
        self.model = Model(
            in_size=self.encoding.dim,
            out_size=len(self.data['train'].get_label_keys()),
            config=self.config['model']['neural']
        ).to(get_device())

        # trainer
        self.trainer = Trainer(
            self.model,
            self.data,
            self.collation_fn,
            out_dir=self.config['out_path'],
            config=self.config['trainer'],
        )

    #  -------- __call__ -----------
    #
    def __call__(self):
        self.trainer()

    #
    #
    #  -------- collation_fn -----------
    #
    def collation_fn(self, batch: list) -> tuple:
        text: list = []
        label: list = []

        # collate data
        for _, review, sentiment in batch:
            text.append(review)
            label.append(sentiment)

        # embed text
        _, sent_embeds, _ = self.encoding(text)

        # extract only first embeddings (CLS); transform labels
        return (
            sent_embeds[:, 1],
            torch.tensor(
                [self.data['train'].encode_label(lb) for lb in label],
                dtype=torch.long, device=get_device()
            )
        )

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self) -> dict:
        return {
            name: Data(
                data_path=path,
                polarities=self.config['data']['polarities'],
                data_label=self.config['data']['data_label'],
                target_label=self.config['data']['target_label']
            ) for name, path in self.config['data']['paths'].items()
        }

    #
    #
    #  -------- __load_config -----------
    #
    @staticmethod
    def __load_config() -> dict:
        # get console arguments, config file
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-C",
            dest="config",
            nargs='+',
            required=True,
            help="define multiple config.json files witch are combined during runtime (overwriting is possible)"
        )
        args = parser.parse_args()

        config_collection: dict = {}

        for config in args.config:
            dict_merge(config_collection, load_json(config))

        return config_collection

    #
    #
    #  -------- __load_logger -----------
    #
    @staticmethod
    def __load_logger(path: str, debug: bool = False) -> None:
        logging.basicConfig(
            level=logging.INFO if not debug else logging.DEBUG,
            format="%(message)s",
            handlers=[
                logging.FileHandler(path, mode="w"),
                logging.StreamHandler()
            ]
        )
        logging.info(f'> Loaded logger: {path}')

    #  -------- __setup_pytorch -----------
    #
    @staticmethod
    def __setup_pytorch(seed: int, cuda: int) -> None:

        # check if cuda is available
        if not torch.cuda.is_available():
            cuda = None

        else:
            # set cuda to last device
            if cuda == -1 or cuda > torch.cuda.device_count():
                cuda = torch.cuda.device_count() - 1

        # make pytorch computations deterministic
        logging.info(f'> Setup PyTorch: seed({seed}), cuda({cuda})')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(cuda) if cuda else None


if __name__ == '__main__':
    Main()()
