import argparse

import torch
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(x)
        return self.decoder(bottleneck), bottleneck

class Encoder(nn.Module):
    def __init__(self, num_features, num_hidden):
        super().__init__()
        self.distribution: NormalDistribution = NormalDistribution()
        self._shared_gcn = GraphConv(
            num_features,
            num_hidden,
            activation=nn.ReLU()
        )
        self.encoder = {
            "log_sigma": nn.Sequential(
                self._shared_gcn,
                GraphConv(num_hidden, num_hidden)
            ),
            "mu": nn.Sequential(
                self._shared_gcn,
                GraphConv(num_hidden, num_hidden)
            )
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distribution =  NormalDistribution(
            mu          = self.encoder["mu"](x),
            log_sigma   = self.encoder["log_sigma"](x)
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
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.encoder = Encoder(args.num_features, args.num_hidden)
        self.decoder = Decoder()

    def distribution(self) -> NormalDistribution:
        return self.decoder.distribution