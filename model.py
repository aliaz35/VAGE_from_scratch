import argparse

import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import Sequential

from abc import ABC, abstractmethod

from attr import dataclass


@dataclass
class NormalDistribution:
    mu : torch.Tensor = 0
    log_sigma : torch.Tensor = 0

class VAE(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @staticmethod
    def sample_latent(p: NormalDistribution) -> torch.Tensor:
        noise = torch.rand_like(p.log_sigma)
        return p.mu + noise * torch.exp(p.log_sigma)

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(*args)
        return self.decoder(bottleneck), bottleneck

class Encoder(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.distribution: NormalDistribution = NormalDistribution()
        self._shared_gcn = GraphConv(
            in_feats,
            out_feats,
            activation=nn.ReLU()
        )
        self.encoder = {
            "log_sigma": Sequential(
                self._shared_gcn,
                GraphConv(out_feats, out_feats)
            ),
            "mu": Sequential(
                self._shared_gcn,
                GraphConv(out_feats, out_feats)
            )
        }

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        distribution =  NormalDistribution(
            mu          = self.encoder["mu"](graph, feat),
            log_sigma   = self.encoder["log_sigma"](graph, feat)
        )
        self.distribution = distribution

        return VAE.sample_latent(distribution)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(bottleneck @ bottleneck.t())


class VGAE(VAE):
    def __init__(self, in_feats: int, hidden_feats: int):
        super().__init__()
        self.encoder = Encoder(in_feats=in_feats, out_feats=hidden_feats)
        self.decoder = Decoder()

    def distribution(self) -> NormalDistribution:
        return self.encoder.distribution