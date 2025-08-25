import torch
import torch.nn as nn
import yaml
from models.attention import SelfAttentionLayer, DecoderBlock
from models.mlp import MLP
from dataclasses import dataclass
import math

@dataclass
class HiddenState:
    z_L: torch.Tensor
    z_H: torch.Tensor

    def detach(self):
        return HiddenState(z_L=self.z_L.detach(), z_H=self.z_H.detach())


class HHRM(nn.Module):

    def __init__(self, config):
        super().__init__()

        if type(config) is str:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        self.name = self.config['name']
        self.num_node_features = self.config['num_node_features']
        self.num_edges = self.config['num_edges']
        self.hidden_dim = self.config['hidden_dim']

        ### Hierarchical reasoning parameters
        hrm_cfg = self.config['hrm']
        self.iters_L = hrm_cfg['iters_L']
        self.iters_H = hrm_cfg['iters_H']
        self.segments = hrm_cfg['segments']

        ### Initial, static hidden states
        self.register_buffer("z_L_init", torch.nn.init.trunc_normal_(torch.empty(1, self.hidden_dim)))
        self.register_buffer("z_H_init", torch.nn.init.trunc_normal_(torch.empty(1, self.hidden_dim)))

        ### Node feature embedding
        emb_cfg = self.config['node_embedder']
        self.node_embedder = MLP(
            input_dim=self.num_node_features,
            layers=emb_cfg['layers'],
            output_dim=emb_cfg['output_dim'],
            activation=emb_cfg['activation']
        )

        ### Edge positional embedding
        self.edge_embedder = nn.Embedding(self.num_edges, self.hidden_dim)
        
        ### Injection layer norm
        self.norm_L = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
        self.norm_H = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

        ### Nodes (low-level) updated based on hyperedges (high-level)
        CA_L_cfg = self.config['node_CA_L']
        self.CA_L = nn.ModuleList([
                            DecoderBlock(
                                model_dim=CA_L_cfg['model_dim'],
                                num_heads=CA_L_cfg['num_heads'],
                                activation=CA_L_cfg['activation'],
                                gated=CA_L_cfg['gated'],
                                scaling=CA_L_cfg['scaling'],
                                ffn_factor=CA_L_cfg['ffn_factor'],
                            )
                            for _ in range(CA_L_cfg['num_layers'])
                        ])
        
        ### Hyperedges (high-level) updated based on nodes (low-level)
        CA_H_cfg = self.config['edge_CA_H']
        self.CA_H = nn.ModuleList([
                            DecoderBlock(
                                model_dim=CA_H_cfg['model_dim'],
                                num_heads=CA_H_cfg['num_heads'],
                                activation=CA_H_cfg['activation'],
                                gated=CA_H_cfg['gated'],
                                scaling=CA_H_cfg['scaling'],
                                ffn_factor=CA_H_cfg['ffn_factor'],
                            )
                            for _ in range(CA_H_cfg['num_layers'])
                        ])

        ### Incidence predictor
        # inc_pred_cfg = self.config['incidence_predictor']
        # self.incidence_predictor = MLP(
        #             input_dim=inc_pred_cfg['input_dim'],
        #             layers=inc_pred_cfg['layers'],
        #             output_dim=inc_pred_cfg['output_dim'],
        #             activation=inc_pred_cfg['activation']
        # )

        ### Indicator predictor
        ind_pred_cfg = self.config['indicator_predictor']
        self.indicator_predictor = MLP(
                    input_dim=ind_pred_cfg['input_dim'],
                    layers=ind_pred_cfg['layers'],
                    output_dim=ind_pred_cfg['output_dim'],
                    activation=ind_pred_cfg['activation']
        )

    def get_init_state(self):
        return HiddenState(z_L=self.z_L_init, z_H=self.z_H_init)

    def forward(self, hid_state: HiddenState, input_state: torch.Tensor):

        z_L = hid_state.z_L
        z_H = hid_state.z_H

        ### key_padding_mask
        node_mask = torch.isnan(input_state).any(dim=-1)
        if node_mask.any():
            input_state = torch.nan_to_num(input_state, nan=0.0)
        else:
            node_mask = None

        ### Input embedding
        if input_state.shape[-1] == self.num_node_features:
            input_state = self.node_embedder(input_state)

        ### Hyperedge positional embedding
        edge_pos_idx = torch.arange(self.num_edges, device=input_state.device).unsqueeze(0).expand(input_state.shape[0], -1)
        edge_pos_emb = self.edge_embedder(edge_pos_idx)
        z_H = z_H + edge_pos_emb

        ### Forward up to last iteration
        with torch.no_grad():
            for iter_H in range(self.iters_H):
                last_iter_H = (iter_H == self.iters_H - 1)

                for iter_L in range(self.iters_L):
                    last_iter_L = (iter_L == self.iters_L - 1)

                    if not (last_iter_H and last_iter_L):
                        ### Low-level update
                        z_L = self.norm_L(z_L + input_state)
                        for block in self.CA_L:
                            z_L = block(z_L, z_H, key_padding_mask_SA=node_mask)

                if not last_iter_H:
                    ### High-level update
                    z_H = self.norm_H(z_H + edge_pos_emb)
                    for block in self.CA_H:
                        z_H = block(z_H, z_L, key_padding_mask_CA=node_mask)

        assert not z_H.requires_grad and not z_L.requires_grad

        ### 1-step gradient approximation
        z_L = self.norm_L(z_L + input_state)
        for block in self.CA_L:
            z_L = block(z_L, z_H, key_padding_mask_SA=node_mask)
        z_H = self.norm_H(z_H + edge_pos_emb)
        for block in self.CA_H:
            z_H = block(z_H, z_L, key_padding_mask_CA=node_mask)

        ### prediction
        # inc = self.incidence_predictor(z_H) # (B, N, K)
        # ind = self.indicator_predictor(z_H.mean(dim=1, keepdim=True)) # (B, 1, K)
        # im = torch.cat([inc, ind], dim=1) # (B, N+1, K)
        # im = im.transpose(1, 2)
        inc = (z_H @ torch.transpose(z_L, 1, 2)) / math.sqrt(self.hidden_dim) # (B, K, N)
        ind = self.indicator_predictor(z_H) # (B, K, 1)
        im = torch.cat([inc, ind], dim=2) # (B, K, N+1)

        ### new state
        state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach())

        return im, state
