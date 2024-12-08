import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch import GraphConv

from attr import dataclass

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
            activation=nn.ReLU()
        )
        self.mu_gcn = GraphConv(
            hidden_feats,
            out_feats,
            weight=True,
        )
        self.log_sigma_gcn = GraphConv(
            hidden_feats,
            out_feats,
            weight=True,
        )

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self._shared_gcn(graph, feat)
        self.distribution =  NormalDistribution(
            mu          = self.mu_gcn(graph, h),
            log_sigma   = self.log_sigma_gcn(graph, h)
        )

        return sample_latent(self.distribution)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(bottleneck @ bottleneck.t())

class VGAE(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        self.encoder = Encoder(in_feats=in_feats, hidden_feats=hidden_feats, out_feats=out_feats)
        self.decoder = Decoder()

    def distribution(self) -> NormalDistribution:
        return self.encoder.distribution


    def forward(self, graph, feat) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(graph, feat)
        return self.decoder(bottleneck), bottleneck
