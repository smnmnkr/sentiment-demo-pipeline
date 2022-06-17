import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from torch import optim

from classifier import Data, Metric
from classifier.util import load_iterator


@dataclass
class Trainer:
    model: torch.nn.Module
    data: Dict[str, Data]
    collation_fn: Callable
    out_dir: str
    config: dict = None

    #  -------- __post_init__ -----------
    #
    def __post_init__(self):

        self.state: dict = {
            'epoch': [],
            'loss_train': [],
            'loss_eval': [],
            'f1_train': [],
            'f1_eval': [],
            'duration': [],
        }

        # load default config file if is None
        if self.config is None:
            self.config = self.default_config

        # setup loss_fn, optimizer, scheduler and early stopping
        self.metric = Metric()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), **self.config['optimizer'])

    #  -------- default_config -----------
    #
    @property
    def default_config(self) -> dict:
        return {
            "seed": 1,
            "epochs": 25,
            "shuffle": True,
            "batch_size": 32,
            "num_workers": 0,
            "report_rate": 1,
            "max_grad_norm": 1.0,
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "betas": [
                    0.9,
                    0.98
                ],
                "eps": 1e-9
            }
        }

    #
    #
    #  -------- __call__ (train) -----------
    #
    def __call__(self) -> dict:
        logging.info(f'\n[--- TRAIN -> {self.data["train"].data_path} ---]')

        saved_model_epoch: int = 0
        saved_eval_metric: tuple = ()

        # --- epoch loop
        try:
            for epoch in range(1, self.config["epochs"] + 1):
                time_begin: datetime = datetime.now()

                # --- ---------------------------------
                # --- begin train
                self.metric.reset()
                loss_train: float = 0.0
                for idx, batch in load_iterator(
                        self.data['train'],
                        collate_fn=self.collation_fn,
                        batch_size=self.config["batch_size"],
                        shuffle=self.config["shuffle"],
                        num_workers=self.config["num_workers"],
                        desc=f"Train, epoch: {epoch:03}",
                        disable=epoch % self.config["report_rate"] != 0
                ):
                    loss_train = self._train(batch, idx, loss_train)
                f1_train: float = self.metric.f_score()

                # --- ---------------------------------
                # --- begin eval
                self.metric.reset()
                loss_eval: float = 0.0
                for idx, batch in load_iterator(
                        self.data['eval'],
                        collate_fn=self.collation_fn,
                        batch_size=self.config["batch_size"],
                        shuffle=self.config["shuffle"],
                        num_workers=self.config["num_workers"],
                        desc=f"Eval, epoch: {epoch:03}",
                        disable=epoch % self.config["report_rate"] != 0
                ):
                    loss_eval = self._eval(batch, idx, loss_eval)
                f1_eval: float = self.metric.f_score()

                # --- ---------------------------------
                # --- update state
                self.state["epoch"].append(epoch)
                self.state["loss_train"].append(loss_train)
                self.state["loss_eval"].append(loss_eval)
                self.state["f1_train"].append(f1_train)
                self.state["f1_eval"].append(f1_eval)
                self.state["duration"].append(datetime.now() - time_begin)

                # --- ---------------------------------
                # --- save if is best model
                if self.state["f1_eval"][-1] >= max(n for n in self.state["f1_eval"] if n > 0):
                    saved_model_epoch = self.state["epoch"][-1]
                    self.model.save(self.out_dir + "model.bin")
                    saved_eval_metric = self.metric.save()

                # --- ---------------------------------
                # --- log to user
                if epoch % self.config["report_rate"] == 0:
                    self._log(epoch)

        except KeyboardInterrupt:
            logging.warning('> Warning: Training interrupted by user!')

        # load last save model
        logging.info('> Load best model based on evaluation loss.')
        self.model = self.model.load(self.out_dir + "model.bin")
        self._log(saved_model_epoch)

        # return and write train state to main
        self._write_state()

        # --- ---------------------------------
        # --- eval
        logging.info(f'\n[--- EVAL -> {self.data["eval"].data_path} ---]')
        self.metric.load(saved_eval_metric)
        self.metric.show(decoding=self.data['train'].decode_label)
        self.metric.export(f'{self.out_dir}metric.eval', decoding=self.data['train'].decode_label)

        return self.state

    #
    #
    #  -------- _train -----------
    #
    def _train(self, batch: tuple, batch_id: int, loss_train: float) -> float:
        self.model.train()

        # zero the parameter gradients
        self.optimizer.zero_grad()

        loss, pred_labels = self.model.train_step(self.loss_fn, batch)
        loss.backward()

        # scaling the gradients down, places a limit on the size of the parameter updates
        # https://pytorch.org/docs/stable/nn.html#clip-grad-norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])

        # optimizer step
        self.optimizer.step()

        # save loss, acc for statistics
        loss_train += (loss.item() - loss_train) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        # calculate metric
        self._metric(batch, pred_labels)

        return loss_train

    #
    #
    #  -------- _eval -----------
    #
    def _eval(self, batch: tuple, batch_id: int, loss_eval: float) -> float:
        self.model.eval()

        loss, pred_labels = self.model.train_step(self.loss_fn, batch)

        # save loss, acc for statistics
        loss_eval += (loss.item() - loss_eval) / (batch_id + 1)

        # reduce memory usage by deleting loss after calculation
        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
        del loss

        # calculate metric
        self._metric(batch, pred_labels)

        return loss_eval

    #
    #
    #  -------- _metric -----------
    #
    def _metric(self, batch: Tuple[list, torch.Tensor], pred_labels: torch.Tensor) -> None:
        _, gold_labels = batch

        self.metric.confusion_matrix(
            self.data['train'].get_label_values(),
            pd.Series(gold_labels.cpu().numpy()), pd.Series(pred_labels.cpu().numpy())
        )

    #  -------- _log -----------
    #
    def _log(self, epoch: int) -> None:
        logging.info((
            f"@{epoch:03}: \t"
            f"loss(train)={self.state['loss_train'][epoch - 1]:2.4f} \t"
            f"loss(eval)={self.state['loss_eval'][epoch - 1]:2.4f} \t"
            f"f1(train)={self.state['f1_train'][epoch - 1]:2.4f} \t"
            f"f1(eval)={self.state['f1_eval'][epoch - 1]:2.4f} \t"
            f"duration(epoch)={self.state['duration'][epoch - 1]}"
        ))

    #  -------- _write_state -----------
    #
    def _write_state(self) -> None:
        cols: list = list(self.state.keys())

        with open(self.out_dir + 'train.csv', 'w') as output_file:
            writer = csv.writer(output_file, delimiter=',')
            writer.writerow(cols)
            writer.writerows(zip(*[self.state[c] for c in cols]))
