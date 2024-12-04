import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

from dataloader import DataLoader
from input_data import load_data
from preprocessing import *
import model

import dgl
import torch
import argparse
import torch.nn as nn
import random

from torch.nn import BCELoss
from torch.optim import Adam
from model import VGAE
from utils import parse_cmdline
from sklearn.metrics import roc_auc_score, average_precision_score

class _KLDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        pass


class Loss(nn.Module):
    def __init__(self, adj: torch.Tensor):
        super().__init__()
        self.bce_norm = adj.shape[0] * adj.shape[0] / (adj.shape[0] * adj.shape[0] - adj.sum()) * 2
        pos_weight = adj.shape[0] * adj.shape[0] - adj.sum() / adj.sum()
        self.bce_weight = torch.tensor([1 if e == 0 else pos_weight for e in adj.to_dense().view(-1)])
        self.kl_norm = 1 / adj.shape[0]
        self._bce_loss = BCELoss(weight=self.bce_weight)
        self._kl_divergence = \
            lambda mu, log_sigma: 0.5 * (1 + 2 * log_sigma - mu**2 - torch.exp(log_sigma)**2).sum(1).mean()

    def forward(self,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                mu: torch.Tensor,
                log_sigma: torch.Tensor
                ) -> torch.Tensor:
        return (self.bce_norm * self._bce_loss(predictions, labels)
                - self.kl_norm * self._kl_divergence(mu, log_sigma))

class RunProxy:
    def __init__(self, args: argparse.Namespace) -> None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        self.dataloader = DataLoader(args)
        self.model  = VGAE(in_feats=self.dataloader.features.shape[1],
                           hidden_feats=args.hidden_feats).to(args.device)
        self.loss = Loss(self.dataloader.adj).to(args.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.train_loss = None
        self.auc = {}
        self.ap = {}

    def _evaluate(self, edges, A_hat: torch.Tensor) -> tuple[float, float]:
        y_score = []
        y_true = []
        for e in edges:
            y_score.append(A_hat[*e])
            y_true.append(self.dataloader.adj[*e])

        return roc_auc_score(y_true, y_score),\
            average_precision_score(y_true, y_score)

    def train(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.train()
        A_hat, bottleneck = self.model(self.dataloader.train_graph, self.dataloader.features)
        d = self.model.distribution()
        loss = self.loss(A_hat.view(-1),
                         self.dataloader.adj.to_dense().view(-1).float(),
                         d.mu,
                         d.log_sigma)
        self.train_loss = loss
        loss.backward()
        self.optimizer.step()
        return A_hat, bottleneck

    @torch.no_grad()
    def valid(self, A_hat: torch.Tensor) -> None:
        self.model.eval()
        auc, ap = self._evaluate(self.dataloader.valid_edges, A_hat)
        self.auc["valid"] = auc
        self.ap["valid"] = ap

    @torch.no_grad()
    def test(self, A_hat: torch.Tensor) -> None:
        self.model.eval()
        auc, ap = self._evaluate(self.dataloader.test_edges, A_hat)
        self.auc["test"] = auc
        self.ap["test"] = ap


if __name__ == "__main__":
    args = parse_cmdline()
    proxy = RunProxy(args)
    A_hat = None
    bottleneck = None
    for _ in range(0, 200):
        A_hat, bottleneck = proxy.train()
        proxy.valid(A_hat)
        print(f"train_loss: {proxy.train_loss}, valid auc: {proxy.auc["valid"]}, valid ap: {proxy.ap["valid"]}")
    proxy.test(A_hat)
    print(f"test_auc: {proxy.auc["test"]}, test_ap: {proxy.ap["test"]}")
