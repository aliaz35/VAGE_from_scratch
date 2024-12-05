import argparse

import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import Sequential

from abc import ABC, abstractmethod

from attr import dataclass
from networkx.classes.filters import hide_nodes


@dataclass
class NormalDistribution:
    mu : torch.Tensor = 0
    log_sigma : torch.Tensor = 0

def sample_latent(p: NormalDistribution) -> torch.Tensor:
    noise = torch.randn_like(p.log_sigma)
    return p.mu + noise * torch.exp(p.log_sigma)

class Encoder(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        self.distribution: NormalDistribution = NormalDistribution()
        self._shared_gcn = GraphConv(
            in_feats,
            hidden_feats,
            weight=True,
            norm="left",
            activation=nn.ReLU()
        )
        self.encoder = {
            "log_sigma": Sequential(
                self._shared_gcn,
                GraphConv(
                    hidden_feats,
                    out_feats,
                    weight=True,
                    norm="left",
                )
            ),
            "mu": Sequential(
                self._shared_gcn,
                GraphConv(
                    hidden_feats,
                    out_feats,
                    weight=True,
                    norm="left"
                )
            )
        }

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        self.distribution =  NormalDistribution(
            mu          = self.encoder["mu"](graph, feat),
            log_sigma   = self.encoder["log_sigma"](graph, feat)
        )

        return sample_latent(self.distribution)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        bottleneck = bottleneck / bottleneck.norm(dim=1, keepdim=True)
        return self.sigmoid(bottleneck @ bottleneck.t())

class VGAE(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        self.encoder = Encoder(in_feats=in_feats, hidden_feats=hidden_feats, out_feats=out_feats)
        self.decoder = Decoder()

    def distribution(self) -> NormalDistribution:
        return self.encoder.distribution


    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(*args)
        return self.decoder(bottleneck), bottleneck
