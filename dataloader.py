import torch
import numpy as np
from abc import abstractmethod, ABC
import pickle as pkl
import networkx as nx

class Load(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, dataset: str, extension: str):
        pass

class DirectLoad(Load):
    def __init__(self):
        super().__init__()

    def load(self, dataset: str, extension: str):
        with open(f"data/ind.{dataset}.{extension}", "r") as f:
            return [int(line) for line in f]

class PklLoad(Load):
    def __init__(self):
        super().__init__()

    def load(self, dataset: str, extension: str):
        with open(f"data/ind.{dataset}.{extension}", "rb") as f:
            return pkl.load(f, encoding="latin1")

class LoadX(PklLoad):
    def __init__(self):
        super().__init__()

    def _load_wrapper(self, dataset: str, extension: str) -> torch.Tensor:
        x = self.load(dataset, extension)
        return torch.sparse_coo_tensor(np.array(x.nonzero()), x.data, x.shape)

    def __call__(
            self,
            dataset: str,
    ) -> dict[str, torch.sparse_coo_tensor]:
        return {
            "x"     : self._load_wrapper(dataset, "x"),
            "tx"    : self._load_wrapper(dataset, "tx"),
            "allx"  : self._load_wrapper(dataset, "allx"),
        }

class LoadAdj(PklLoad):
    def __init__(self):
        super().__init__()

    def __call__(self, dataset: str) -> torch.sparse_coo_tensor:
        A = nx.adjacency_matrix(nx.from_dict_of_lists(self.load(dataset, "graph")))
        return {"A" : torch.sparse_coo_tensor(
            np.array(A.nonzero()),
            A.data,
            A.shape
        )}

class LoadIndices(DirectLoad):
    def __init__(self):
        super().__init__()

    def __call__(self, dataset: str, extension: str) -> list[int]:
        return self.load(dataset, extension)

class Dataset(LoadX, LoadAdj):
    def __init__(self, dataset: str):
        super().__init__()
        self._data = LoadX.__call__(self, dataset) | LoadAdj.__call__(self, dataset)

    def extract(self):
        return self._data