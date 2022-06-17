import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Set

import pandas as pd


@dataclass
class Metric:
    beta: float = 1.0
    _tps: defaultdict = field(default_factory=defaultdict)
    _fps: defaultdict = field(default_factory=defaultdict)
    _tns: defaultdict = field(default_factory=defaultdict)
    _fns: defaultdict = field(default_factory=defaultdict)

    #  -------- __post_init__ -----------
    #
    def __post_init__(self) -> None:
        self.reset()

    #  -------- reset -----------
    #
    def reset(self) -> None:
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    #
    #
    #  -------- confusion_matrix -----------
    #
    def confusion_matrix(
            self, classes: Set[str],
            golds: pd.Series, preds: pd.Series,
    ):

        for cls in classes:
            # create confusing matrix values for each category (omitting true negative)
            tp: int = sum(pd.Series((preds == golds) & (golds == cls)))

            self.add(
                cls, tps=tp,
                fns=sum(pd.Series(golds == cls)) - tp,
                fps=sum(pd.Series(preds == cls)) - tp
            )

            # add for every other class category matches to true negative
            for nc in (classes - {cls}):
                self.add(nc, tns=tp)

    #
    #
    #  -------- precision -----------
    #
    def precision(self, class_name: str = None):

        # get tp, fn
        tp: float = self.get_tp(class_name)
        fp: float = self.get_fp(class_name)

        if sum([tp, fp]) > 0:
            return (
                    tp / (tp + fp)
            )

        return 0.0

    #
    #
    #  -------- recall -----------
    #
    def recall(self, class_name: str = None):

        # get tps, fns
        tp: float = self.get_tp(class_name)
        fn: float = self.get_fn(class_name)

        if sum([tp, fn]) > 0:
            return (
                    tp / (tp + fn)
            )

        return 0.0

    #
    #
    #  -------- f_score -----------
    #
    def f_score(self, class_name: str = None):

        # get precision, recall, beta squared
        pre: float = self.precision(class_name)
        rec: float = self.recall(class_name)
        b2: float = self.beta ** 2

        if sum([pre, rec]) > 0:
            return (
                    (1 + b2) * ((pre * rec) / ((pre * b2) + rec))
            )

        return 0.0

    #
    #
    #  -------- accuracy -----------
    #
    def accuracy(self, class_name: str = None):

        # get tp, fp, fn, tn
        tp: float = self.get_tp(class_name)
        fp: float = self.get_fp(class_name)
        fn: float = self.get_fn(class_name)
        tn: float = self.get_tn(class_name)

        if sum([tp, fp, fn, tn]) > 0:
            return (
                    (tp + tn) / sum([tp, fp, fn, tn])
            )

        return 0.0

    #
    #
    #  -------- get_classes -----------
    #
    def get_classes(self) -> list:

        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._fns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )

        all_classes = [
            class_name
            for class_name in all_classes
            if class_name is not None
        ]

        all_classes.sort()
        return all_classes

    #
    #
    #  -------- show -----------
    #
    def show(
            self,
            decoding: callable = None,
    ):
        def dec(n): return decoding(n) if decoding else n

        for class_name in [None] + self.get_classes():
            logging.info((
                (f"{'AVG' if class_name is None else dec(class_name):14}"
                 f"\t tp: {self.get_tp(class_name):8}"
                 f"\t fp: {self.get_fp(class_name):8} "
                 f"\t tn: {self.get_tn(class_name):8}"
                 f"\t fn: {self.get_fn(class_name):8}"
                 f"\t pre={self.precision(class_name):2.4f}"
                 f"\t rec={self.recall(class_name):2.4f}"
                 f"\t f1={self.f_score(class_name):2.4f}"
                 f"\t acc={self.accuracy(class_name):2.4f}")
            ))

    #  -------- pass -----------
    #
    def export(self, path: str, decoding: callable = None) -> None:
        def dec(n): return decoding(n) if decoding else n

        pd.DataFrame([
            [
                'AVG' if class_name is None else dec(class_name),
                self.get_tp(class_name),
                self.get_fp(class_name),
                self.get_tn(class_name),
                self.get_fn(class_name),
                self.precision(class_name),
                self.recall(class_name),
                self.f_score(class_name),
                self.accuracy(class_name)
            ]
            for class_name in [None] + self.get_classes()
        ], columns=['label', 'tp', 'fp', 'tn', 'fn', 'prec', 'rec', 'f1', 'acc']
        ).to_csv(f'{path}.csv', index=False)

    #  -------- save -----------
    #
    def save(self) -> Tuple[dict, dict, dict, dict]:
        return self._tps, self._fps, self._tns, self._fns

    #  -------- load -----------
    #
    def load(self, data: Tuple[dict, dict, dict, dict]) -> None:
        self._tps, self._fps, self._tns, self._fns = data

    #  -------- add_tp -----------
    #
    def add(self,
            class_name: str,
            tps: int = 0,
            tns: int = 0,
            fps: int = 0,
            fns: int = 0
            ):
        self.add_tp(class_name, tps)
        self.add_tn(class_name, tns)
        self.add_fp(class_name, fps)
        self.add_fn(class_name, fns)

    #  -------- add_tp -----------
    #
    def add_tp(self, class_name: str, amount: int = 1):
        self._tps[class_name] += amount

    #  -------- add_tp -----------
    #
    def add_tn(self, class_name: str, amount: int = 1):
        self._tns[class_name] += amount

    #  -------- add_fp -----------
    #
    def add_fp(self, class_name: str, amount: int = 1):
        self._fps[class_name] += amount

    #  -------- add_fn -----------
    #
    def add_fn(self, class_name: str, amount: int = 1):
        self._fns[class_name] += amount

    #  -------- _get -----------
    #
    def _get(self, cat: dict, class_name: str = None):
        if class_name is None:
            return sum(
                [cat[class_name] for class_name in self.get_classes()]
            )
        return cat[class_name]

    #  -------- get_tp -----------
    #
    def get_tp(self, class_name: str = None):
        return self._get(self._tps, class_name)

    #  -------- get_tn -----------
    #
    def get_tn(self, class_name: str = None):
        return self._get(self._tns, class_name)

    #  -------- get_fp -----------
    #
    def get_fp(self, class_name: str = None):
        return self._get(self._fps, class_name)

    #  -------- get_fn -----------
    #
    def get_fn(self, class_name: str = None):
        return self._get(self._fns, class_name)

    #  -------- get_actual -----------
    #
    def get_actual(self, class_name: str = None):
        return self.get_tp(class_name) + self.get_fn(class_name)

    #  -------- get_predicted -----------
    #
    def get_predicted(self, class_name: str = None):
        return self.get_tp(class_name) + self.get_fp(class_name)
