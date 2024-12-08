import argparse
from dataclasses import dataclass

import torch
import dgl
import numpy as np
import scipy
import pickle as pkl
import networkx as nx
import random

def _graph_wrapper(x) -> torch.Tensor:
    A = nx.adjacency_matrix(nx.from_dict_of_lists(x))
    return torch.sparse_coo_tensor(
        np.array(A.nonzero()),
        A.data,
        A.shape
    )

@dataclass
class Dataset:
    adj: torch.Tensor
    features: torch.Tensor
    train_graph: dgl.DGLGraph
    valid_edges: torch.Tensor
    test_edges: torch.Tensor

class DataLoader:
    def __init__(self, args: argparse.Namespace):
        self.dataset = args.dataset
        self.split_ratio = args.split_ratio

    def load(self) -> Dataset:
        _raw_data = self.read_data()
        features = self.load_features(_raw_data)
        adj = self.load_adj(_raw_data)
        _sampled_edges = self._sample_edge(adj)
        train_graph = dgl.to_bidirected(dgl.graph(
            tuple(zip(*_sampled_edges["train"])),
            num_nodes=adj.shape[0]
        ).add_self_loop())
        valid_edges = _sampled_edges["valid"]
        test_edges = _sampled_edges["test"]
        return Dataset(
            adj=adj,
            features=features,
            train_graph=train_graph,
            valid_edges=valid_edges,
            test_edges=test_edges
        )

    def read_data(self) -> dict:
        return {
            "x": self.pkl_read(
                self.dataset,
                "x",
                lambda x: torch.sparse_coo_tensor(
                    np.array(x.nonzero()),
                    x.data,
                    x.shape
                )
            ),
            "tx": self.pkl_read(
                self.dataset,
                "tx",
                lambda x: torch.sparse_coo_tensor(
                    np.array(x.nonzero()),
                    x.data,
                    x.shape
                )
            ),
            "allx": self.pkl_read(
                self.dataset,
                "allx",
                lambda x: torch.sparse_coo_tensor(
                    np.array(x.nonzero()),
                    x.data,
                    x.shape
                )
            ),
            "graph": self.pkl_read(
                self.dataset,
                "graph",
                _graph_wrapper
            ),
            "index": self.direct_read(
                self.dataset,
                "test.index"
            )
        }

    def _sample_edge(self, adj: torch.Tensor):
        sampled_edges = self._split_edges(adj
                 .to_dense()
                 .triu(1)
                 .nonzero())

        return {
            "train" : sampled_edges["train"],
            "valid" : torch.vstack((sampled_edges["valid"], self._sample_negative(adj, len(sampled_edges["valid"])))),
            "test": torch.vstack((sampled_edges["test"], self._sample_negative(adj, len(sampled_edges["test"])))),
        }

    def _split_edges(self, edges: torch.Tensor) -> dict[str, torch.Tensor]:
        ratios = list(map(lambda x: int(x) / 100, self.split_ratio.split(':')))
        assert sum(ratios) == 1 and len(ratios) == 3

        ret = {}
        random_indices = list(range(len(edges)))
        random.shuffle(random_indices)

        test_end = int(ratios[-1] * len(edges))
        ret["test"] = edges[random_indices[:test_end]]

        valid_end = int(ratios[-2] * len(edges)) + test_end
        ret["valid"] = edges[random_indices[test_end:valid_end]]

        ret["train"] = edges[random_indices[valid_end:]]

        return ret

    def _sample_negative(self, adj: torch.Tensor, length: int) -> torch.Tensor:
        ret = set()
        while len(ret) < length:
            u = random.randrange(0, adj.shape[0])
            v = random.randrange(0, adj.shape[0])
            if (u == v
                    or (u, v) in ret
                    or (v, u) in ret
                    or adj[u][v] == 1):
                continue
            ret.add((u, v))

        return torch.tensor(list(ret))

    @staticmethod
    def direct_read(
            dataset: str,
            extension: str
    ) -> list[int]:
        with open(f"data/ind.{dataset}.{extension}", "r") as f:
            return [int(line) for line in f]

    @staticmethod
    def pkl_read(
            dataset: str,
            extension: str,
            wrapper: callable=None
    ) ->scipy.sparse.csr_matrix | torch.Tensor:
        with open(f"data/ind.{dataset}.{extension}", "rb") as f:
            ret = pkl.load(f, encoding="latin1")
            return ret if wrapper is None else wrapper(ret)

    @staticmethod
    def load_features(raw_data: dict[str, torch.Tensor]) -> torch.Tensor:
        tx = raw_data["tx"]
        indices = raw_data["index"]
        ret = torch.zeros(max(indices) + 1, tx.shape[1])
        ret[range(min(indices))] = raw_data["allx"].to_dense()
        ret[indices] = raw_data["tx"].to_dense()
        return ret.to_sparse()

    @staticmethod
    def load_adj(raw_data: dict[str, torch.Tensor]) -> torch.Tensor:
        return raw_data["graph"]