import argparse

import torch
import dgl
import numpy as np
import scipy
import pickle as pkl
import networkx as nx
import random

from sympy.core.random import sample


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

class DataLoader:
    def __init__(self, args: argparse.Namespace):
        # def __init__(self, dataset: str):
        self.args = args
        raw_data = DataLoader.read_files(args.dataset)
        self.adj, self.features = self.process_raw_data(raw_data)
        sampled_edges = self._sample_edge()
        self.train_graph = dgl.to_bidirected(dgl.graph(
            tuple(zip(*sampled_edges["train"])),
            num_nodes=self.adj.shape[0]
        ).add_self_loop())
        self.valid_edges = sampled_edges["valid"]
        self.test_edges = sampled_edges["test"]

    @staticmethod
    def process_raw_data(
            raw_data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tx = raw_data["tx"]
        indices = raw_data["index"]
        features = torch.zeros(max(indices) + 1, tx.shape[1])
        features[range(min(indices))] = raw_data["allx"].to_dense()
        features[indices] = raw_data["tx"].to_dense()
        return raw_data["graph"], features.to_sparse()

    def _sample_edge(self):
        sampled_edges = self._split_edges(self.adj
                 .to_dense()
                 .triu(1)
                 .nonzero())

        return {
            "train" : sampled_edges["train"],
            "valid" : torch.vstack((sampled_edges["valid"], self._sample_negative(len(sampled_edges["valid"])))),
            "test": torch.vstack((sampled_edges["test"], self._sample_negative(len(sampled_edges["test"])))),
        }

    def _split_edges(self, edges: torch.Tensor) -> dict[str, torch.Tensor]:
        ratios = list(map(lambda x: int(x) / 100, self.args.split_ratio.split(':')))
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

    def _sample_negative(self, length: int) -> torch.Tensor:
        ret = set()
        while len(ret) < length:
            u = random.randrange(0, self.adj.shape[0])
            v = random.randrange(0, self.adj.shape[0])
            if (u == v
                    or (u, v) in ret
                    or (v, u) in ret
                    or self.adj[u][v] == 1):
                continue
            ret.add((u, v))

        return torch.tensor(list(ret))


    @staticmethod
    def read_files(dataset) -> dict:
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



