import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
# from hypergraphflow.models.transformer import DecoderBlock, ContextProjector
from models.time import TimestepEmbedder
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
        self.timestep_embedding = self.config['timestep_embedding']
        self.output_norm = self.config.get('output_norm', None)

        ### Hierarchical reasoning parameters
        hrm_cfg = self.config['hrm']
        self.iters_L = hrm_cfg['iters_L']
        self.iters_H = hrm_cfg['iters_H']
        self.segments = hrm_cfg['segments']
        self.use_draft = hrm_cfg.get('use_draft', False)

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

        ### Timestep embedding
        if self.timestep_embedding:
            self.timestep_embedder = TimestepEmbedder(
                                        hidden_size=self.config['time_dim'],
                                        frequency_embedding_size=self.config['freq_dim']
                                        )
        else:
            self.timestep_embedder = None

        ### Nodes (low-level) updated based on hyperedges (high-level)
        CA_L_cfg = self.config['node_CA_L']
        # if self.timestep_embedding:
        #     self.context_projector_L = ContextProjector(
        #                                 c_dim=CA_L_cfg['c_dim'],
        #                                 model_dim=CA_L_cfg['model_dim'],
        #                                 gated=CA_L_cfg['gated'],
        #                                 activation=CA_L_cfg['activation']
        #                             )

        # self.CA_L = nn.ModuleList([
        #                     DecoderBlock(
        #                         model_dim=CA_L_cfg['model_dim'],
        #                         attn_type=CA_L_cfg.get('attn_type', 'torch'),
        #                         attn_bias=CA_L_cfg.get('attn_bias', False),
        #                         attn_dropout=CA_L_cfg.get('attn_dropout', 0.0),
        #                         attn_do_qkv_norm=CA_L_cfg.get('attn_do_qkv_norm', True),
        #                         num_heads=CA_L_cfg['num_heads'],
        #                         activation=CA_L_cfg['activation'],
        #                         c_dim=CA_L_cfg['c_dim'] if self.timestep_embedding else None,
        #                         gated=CA_L_cfg['gated'],
        #                         scaling=CA_L_cfg['scaling'],
        #                         ffn_factor=CA_L_cfg['ffn_factor'],
        #                         c_proj=self.context_projector_L if self.timestep_embedding else None,
        #                     )
        #                     for _ in range(CA_L_cfg['num_layers'])
        #                 ])
        
        from models.transformer import DiTDecoder
        self.CA_L = DiTDecoder(**CA_L_cfg)

        ### Hyperedges (high-level) updated based on nodes (low-level)
        CA_H_cfg = self.config['edge_CA_H']
        # if self.timestep_embedding:
        #     self.context_projector_H = ContextProjector(
        #                                 c_dim=CA_H_cfg['c_dim'],
        #                                 model_dim=CA_H_cfg['model_dim'],
        #                                 gated=CA_H_cfg['gated'],
        #                                 activation=CA_H_cfg['activation']
        #                             )
        # self.CA_H = nn.ModuleList([
        #                     DecoderBlock(
        #                         model_dim=CA_H_cfg['model_dim'],
        #                         num_heads=CA_H_cfg['num_heads'],
        #                         activation=CA_H_cfg['activation'],
        #                         c_dim=CA_H_cfg['c_dim'] if self.timestep_embedding else None,
        #                         gated=CA_H_cfg['gated'],
        #                         scaling=CA_H_cfg['scaling'],
        #                         ffn_factor=CA_H_cfg['ffn_factor'],
        #                         c_proj=self.context_projector_H if self.timestep_embedding else None,
        #                     )
        #                     for _ in range(CA_H_cfg['num_layers'])
        #                 ])
        self.CA_H = DiTDecoder(**CA_H_cfg)

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

    def get_time_emb(self, segment: int, iter_L: int, iter_H: int):
        if self.timestep_embedding:
            t_step = iter_L + iter_H * self.iters_L + segment * self.iters_L * self.iters_H
            t_frac = t_step / (self.iters_L * self.iters_H * self.segments)
            t_frac = torch.tensor([t_frac], dtype=torch.float32, device=self.z_L_init.device)
            t_emb = self.timestep_embedder(t_frac)
            # print(f"segment: {segment}, iter_L: {iter_L}, iter_H: {iter_H}, t_step: {t_step}, t_frac: {t_frac.item():.4f}")
            return t_emb
        else:
            return None

    def dot_prod_incidence(self, q, k):
        return (q @ torch.transpose(k, 1, 2)) / math.sqrt(self.hidden_dim) # (B, Nq, Nk)

    def normalize_output(self, im):
        if self.output_norm == 'sigmoid':
            im = torch.sigmoid(im)
        elif self.output_norm == 'softmax':
            im = torch.cat([
                F.softmax(im[..., :-1], dim=1),
                torch.sigmoid(im[..., -1:])
            ], dim=-1)
        elif self.output_norm is not None:
            raise ValueError(f"Unknown output_norm {self.output_norm}")
        return im

    def forward(self, hid_state: HiddenState, input_state: torch.Tensor, segment: int,draft=None):

        z_L = hid_state.z_L
        z_H = hid_state.z_H

        ### Node mask
        node_mask = torch.isnan(input_state).any(dim=-1)
        if node_mask.any():
            input_state = torch.nan_to_num(input_state, nan=0.0)
        else:
            node_mask = None

        ### Edge mask
        if self.use_draft and (draft is not None):
            ind_draft = draft[:, :, -1:] # (B, K, 1)
            edge_mask = (ind_draft < 0.2).squeeze(-1)
        else:
            edge_mask = None

        ### Input embedding
        if input_state.shape[-1] == self.num_node_features:
            input_state = self.node_embedder(input_state)

        ### Hyperedge positional embedding
        edge_pos_idx = torch.arange(self.num_edges, device=input_state.device).unsqueeze(0).expand(input_state.shape[0], -1)
        edge_pos_emb = self.edge_embedder(edge_pos_idx)

        ### Forward up to last iteration
        with torch.no_grad():
            z_H = z_H + edge_pos_emb
            for iter_H in range(self.iters_H):
                last_iter_H = (iter_H == self.iters_H - 1)

                for iter_L in range(self.iters_L):
                    last_iter_L = (iter_L == self.iters_L - 1)

                    if not (last_iter_H and last_iter_L):
                        ### Low-level update
                        z_L = self.norm_L(z_L + input_state)
                        t_emb = self.get_time_emb(segment, iter_L, iter_H)
                        # for block in self.CA_L:
                        z_L = self.CA_L(q=z_L, kv=z_H,
                                        context=t_emb,
                                        q_mask=node_mask,
                                        kv_mask=edge_mask
                                    )

                if not last_iter_H:
                    ### High-level update
                    z_H = self.norm_H(z_H + edge_pos_emb)
                    t_emb = self.get_time_emb(segment, self.iters_L - 1, iter_H)
                    # for block in self.CA_H:
                    z_H = self.CA_H(q=z_H, kv=z_L,
                                    context=t_emb,
                                    q_mask=edge_mask,
                                    kv_mask=node_mask
                                )

        assert not z_H.requires_grad and not z_L.requires_grad

        ### 1-step gradient approximation
        z_L = self.norm_L(z_L + input_state)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1)
        z_L = self.CA_L(q=z_L, kv=z_H,
                        context=t_emb,
                        q_mask=node_mask,
                        kv_mask=edge_mask
                    )

        z_H = self.norm_H(z_H + edge_pos_emb)
        # t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1)
        z_H = self.CA_H(q=z_H, kv=z_L,
                        context=t_emb,
                        q_mask=edge_mask,
                        kv_mask=node_mask
                    )

        ### prediction
        inc = self.dot_prod_incidence(q=z_H, k=z_L) # (B, K, N)
        ind = self.indicator_predictor(z_H) # (B, K, 1)
        im = torch.cat([inc, ind], dim=2) # (B, K, N+1)

        ### normalization
        im = self.normalize_output(im)

        ### new state
        state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach())

        return im, state
