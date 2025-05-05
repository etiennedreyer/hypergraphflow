import torch
import torch.nn as nn
import yaml
from models.attention import SelfAttention, CrossAttention
from models.time import TimestepEmbedder
from models.mlp import MLP

class HGFlow(nn.Module):

    def __init__(self, config):
        super().__init__()

        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

        enc_cfg = self.config['node_encoder']
        self.node_encoder = SelfAttention(
            input_dim_kv=enc_cfg['input_dim'],
            model_dim=enc_cfg['model_dim'],
            num_heads=enc_cfg['num_heads'],
            num_layers=enc_cfg['num_layers'],
            activation=enc_cfg['activation']
        )

        ca_cfg = self.config['cross_attention']

        if self.config['timestep_embedding']:
            self.timestep_embedder = TimestepEmbedder(ca_cfg['model_dim'])
        else:
            self.timestep_embedder = None

        self.cross_attention = CrossAttention(
            input_dim_kv=ca_cfg['model_dim'],
            model_dim=ca_cfg['model_dim'],
            num_heads=ca_cfg['num_heads'],
            num_layers=ca_cfg['num_layers'],
            activation=ca_cfg['activation'],
            c_dim=ca_cfg['model_dim'] \
                if self.timestep_embedder is not None else None,
        )


        pred_cfg = self.config['prediction_mlp']
        self.prediction_mlp = nn.Sequential(
                MLP(
                    input_dim=pred_cfg['input_dim'],
                    layers=pred_cfg['layers'],
                    output_dim=pred_cfg['output_dim'],
                    activation=pred_cfg['activation']
                ),
                nn.Sigmoid()
        )

    def forward(self, n, im_t, t=None):

        node_mask = torch.isnan(n).any(dim=-1)

        ### Node encoding (self-attention)
        n = self.node_encoder(n, key_padding_mask=node_mask)

        ### Hyperedge encoding (incidence-weighted sum of node vectors)
        h = torch.einsum('ben, bnd -> bed', im_t, n)

        ### Timestep embedding
        t = self.timestep_embedder(t) \
            if self.timestep_embedder is not None else None

        ### Node update (cross-attention)
        # q: node features
        # k/v: hyperedge features
        n = self.cross_attention(n, h, c=t)

        ### Prediction
        u = self.prediction_mlp(n)

        return u