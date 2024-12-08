from dataset import Dataset, DataLoader

import dgl
import torch
import argparse
import torch.nn as nn
import random

from torch.nn import BCELoss
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from model import VGAE
from utils import parse_cmdline
from sklearn.metrics import roc_auc_score, average_precision_score

class Loss(nn.Module):
    def __init__(self, train_graph: dgl.DGLGraph):
        super().__init__()
        _g = train_graph.remove_self_loop()
        _all_edges_count = _g.num_nodes() * _g.num_nodes()

        _bce_norm = _all_edges_count / ((_all_edges_count - _g.num_edges()) * 2)
        _bce_weight = torch.ones(_all_edges_count)
        _bce_weight[train_graph.adjacency_matrix().to_dense().view(-1) == 1] \
            *= (_all_edges_count - _g.num_edges()) / _g.num_edges()
        self._bce_loss = lambda predictions, labels: \
            _bce_norm * binary_cross_entropy(
                predictions,
                labels,
                weight=_bce_weight
            )

        _kl_norm = 1 / _g.num_nodes()
        self._kl_divergence = \
            lambda mu, log_sigma: \
                (_kl_norm * 0.5 * (1 + 2 * log_sigma - mu ** 2 - torch.exp(log_sigma) ** 2)
                 .sum(1)
                 .mean())

    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                mu: torch.Tensor,
                log_sigma: torch.Tensor
    ) -> torch.Tensor:
        return (self._bce_loss(predictions, labels)
                - self._kl_divergence(mu, log_sigma))

class ModelTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        self.dataloader = DataLoader(args)
        self.dataset = self.dataloader.load()
        self.model  = VGAE(in_feats=self.dataset.features.shape[1],
                           hidden_feats=args.hidden_feats,
                           out_feats=args.out_feats).to(args.device)
        self.criteria = Loss(self.dataset.train_graph).to(args.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)

        self.loss = None

    def _evaluate(self, edges, A_hat: torch.Tensor) -> tuple[float, float]:
        y_score = []
        y_true = []
        for e in edges:
            y_score.append(A_hat[*e])
            y_true.append(self.dataset.adj[*e])

        return roc_auc_score(y_true, y_score),\
            average_precision_score(y_true, y_score)

    def train_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.train()
        A_hat, bottleneck = self.model(self.dataset.train_graph, self.dataset.features)
        d = self.model.distribution()
        self.optimizer.zero_grad()
        self.loss = self.criteria(A_hat.view(-1),
                                  self.dataset.train_graph.adjacency_matrix().to_dense().view(-1).float(),
                                  d.mu,
                                  d.log_sigma)
        self.loss.backward()
        self.optimizer.step()
        return A_hat, bottleneck

    @torch.no_grad()
    def validate_epoch(self, A_hat: torch.Tensor) -> None:
        self.model.eval()
        auc, ap = self._evaluate(self.dataset.valid_edges, A_hat)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {trainer.loss:.4f} | "
            f"Valid auc: {auc:.4f} | "
            f"Valid ap: {ap:.4f}"
        )

    @torch.no_grad()
    def test_epoch(self, A_hat: torch.Tensor) -> None:
        self.model.eval()
        auc, ap = self._evaluate(self.dataset.test_edges, A_hat)
        print(
            f"Test auc: {auc:.4f} | "
            f"Test ap: {ap:.4f}"
        )

if __name__ == "__main__":
    args = parse_cmdline()
    trainer = ModelTrainer(args)
    A_hat = None
    bottleneck = None
    for epoch in range(0, 200):
        A_hat, bottleneck = trainer.train_epoch()
        trainer.validate_epoch(A_hat)
    trainer.test_epoch(A_hat)