import torch
import numpy as np
import scipy
import pickle as pkl
import networkx as nx
from jedi.inference.gradual.typing import Callable


def direct_load(
        dataset: str,
        extension: str
    ) -> list[int]:
    with open(f"data/ind.{dataset}.{extension}", "r") as f:
        return [int(line) for line in f]

def pkl_load(
        dataset: str,
        extension: str,
        wrapper: callable=None
    ) ->scipy.sparse.csr_matrix | torch.Tensor:
    with open(f"data/ind.{dataset}.{extension}", "rb") as f:
        ret = pkl.load(f, encoding="latin1")
        return ret if wrapper is None else wrapper(ret)


def _graph_wrapper(x) -> torch.Tensor:
    A = nx.adjacency_matrix(nx.from_dict_of_lists(x))
    return torch.sparse_coo_tensor(
        np.array(A.nonzero()),
        A.data,
        A.shape
    )


def load_raw_data(dataset) -> dict:
    return {
        "x": pkl_load(
            dataset,
            "x",
            lambda x: torch.sparse_coo_tensor(
                np.array(x.nonzero()),
                x.data,
                x.shape
            )
        ),
        "tx": pkl_load(
            dataset,
            "tx",
            lambda x: torch.sparse_coo_tensor(
                np.array(x.nonzero()),
                x.data,
                x.shape
            )
        ),
        "allx": pkl_load(
            dataset,
            "allx",
            lambda x: torch.sparse_coo_tensor(
                np.array(x.nonzero()),
                x.data,
                x.shape
            )
        ),
        "graph": pkl_load(
            dataset,
            "graph",
            _graph_wrapper
        ),
        "index": direct_load(
            dataset,
            "test.index"
        )
    }