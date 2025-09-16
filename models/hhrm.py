import torch
import torch.nn as nn
import yaml
from models.attention import DecoderBlock, ContextProjector
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

        ### Hierarchical reasoning parameters
        hrm_cfg = self.config['hrm']
        self.iters_L = hrm_cfg['iters_L']
        self.iters_H = hrm_cfg['iters_H']
        self.segments = hrm_cfg['segments']
        self.use_draft = hrm_cfg.get('use_draft', False)
        self.use_im_attn_mask = hrm_cfg.get('use_im_attn_mask', False)

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
        if self.timestep_embedding:
            self.context_projector_L = ContextProjector(
                                        c_dim=CA_L_cfg['c_dim'],
                                        model_dim=CA_L_cfg['model_dim'],
                                        gated=CA_L_cfg['gated'],
                                        activation=CA_L_cfg['activation']
                                    )

        self.CA_L = nn.ModuleList([
                            DecoderBlock(
                                model_dim=CA_L_cfg['model_dim'],
                                num_heads=CA_L_cfg['num_heads'],
                                activation=CA_L_cfg['activation'],
                                c_dim=CA_L_cfg['c_dim'] if self.timestep_embedding else None,
                                gated=CA_L_cfg['gated'],
                                scaling=CA_L_cfg['scaling'],
                                ffn_factor=CA_L_cfg['ffn_factor'],
                                c_proj=self.context_projector_L if self.timestep_embedding else None,
                            )
                            for _ in range(CA_L_cfg['num_layers'])
                        ])
        
        ### Hyperedges (high-level) updated based on nodes (low-level)
        CA_H_cfg = self.config['edge_CA_H']
        if self.timestep_embedding:
            self.context_projector_H = ContextProjector(
                                        c_dim=CA_H_cfg['c_dim'],
                                        model_dim=CA_H_cfg['model_dim'],
                                        gated=CA_H_cfg['gated'],
                                        activation=CA_H_cfg['activation']
                                    )
        self.CA_H = nn.ModuleList([
                            DecoderBlock(
                                model_dim=CA_H_cfg['model_dim'],
                                num_heads=CA_H_cfg['num_heads'],
                                activation=CA_H_cfg['activation'],
                                c_dim=CA_H_cfg['c_dim'] if self.timestep_embedding else None,
                                gated=CA_H_cfg['gated'],
                                scaling=CA_H_cfg['scaling'],
                                ffn_factor=CA_H_cfg['ffn_factor'],
                                c_proj=self.context_projector_H if self.timestep_embedding else None,
                            )
                            for _ in range(CA_H_cfg['num_layers'])
                        ])

        ### Indicator predictor
        ind_pred_cfg = self.config['indicator_predictor']
        self.indicator_predictor = MLP(
                    input_dim=ind_pred_cfg['input_dim'],
                    layers=ind_pred_cfg['layers'],
                    output_dim=ind_pred_cfg['output_dim'],
                    activation=ind_pred_cfg['activation']
        )

        if self.use_im_attn_mask:
            if self.timestep_embedding:
                ### Context-modulated coef with which to add inc. mat. to CA
                self.im_attn_wgt_L = nn.Linear(self.CA_L[0].CA.c_dim, 1)
                self.im_attn_wgt_H = nn.Linear(self.CA_H[0].CA.c_dim, 1)
            else:
                raise NotImplementedError("use_im_attn_mask requires timestep_embedding")

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

    def convert_key_to_attn_mask(self, key_padding_mask, Nq=None):

        if key_padding_mask is None:
            return None

        bs, Nk = key_padding_mask.shape

        if Nq is None:
            Nq = Nk # Assume self-attention

        mask = torch.zeros((bs, Nq, Nk), device=key_padding_mask.device)

        if key_padding_mask is not None:
            if key_padding_mask.any():
                add_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
                add_mask.masked_fill_(key_padding_mask, float('-inf'))
                mask = mask + add_mask.unsqueeze(1) # broadcast (B, 1, Nk) -> (B, Nq, Nk)

        return mask

    def dot_prod_incidence(self, q, k):
        return (q @ torch.transpose(k, 1, 2)) / math.sqrt(self.hidden_dim) # (B, Nq, Nk)

    def forward(self, hid_state: HiddenState, input_state: torch.Tensor, segment: int,draft=None):

        z_L = hid_state.z_L
        z_H = hid_state.z_H

        ### Node mask
        node_mask = torch.isnan(input_state).any(dim=-1)
        if node_mask.any():
            input_state = torch.nan_to_num(input_state, nan=0.0)
        else:
            node_mask = None

        node_CA_mask_H = self.convert_key_to_attn_mask(node_mask, Nq=self.num_edges)
        if self.use_im_attn_mask and (node_CA_mask_H is None):
            node_CA_mask_H = torch.zeros((input_state.shape[0], self.num_edges, input_state.shape[1]), device=input_state.device)
        node_CA_mask_L = None if node_CA_mask_H is None else node_CA_mask_H.transpose(2, 1) # (B, N, K)
        if self.use_im_attn_mask:
            node_CA_mask_H = node_CA_mask_H.repeat(self.CA_H[0].CA.num_heads, 1, 1) # (h*B, K, N)
            node_CA_mask_L = node_CA_mask_L.repeat(self.CA_L[0].CA.num_heads, 1, 1) # (h*B, N, K)

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
                        t_emb = self.get_time_emb(segment, iter_L, iter_H)
                        for block in self.CA_L:
                            CA_mask = node_CA_mask_L
                            if self.use_im_attn_mask:
                                inc_t = self.dot_prod_incidence(q=z_L, k=z_H)
                                inc_t = torch.cat([inc_t, torch.zeros_like(inc_t).repeat(self.CA_L[0].CA.num_heads - 1, 1, 1)], dim=0) # (h*B, N, K)
                                wgt = self.im_attn_wgt_L(t_emb)
                                CA_mask += wgt*inc_t
                            z_L = block(z_L, z_H,
                                        c=t_emb,
                                        key_padding_mask_SA=node_mask,
                                        # key_padding_mask_CA=edge_mask
                                        attn_mask_CA=CA_mask
                                        )

                if not last_iter_H:
                    ### High-level update
                    z_H = self.norm_H(z_H + edge_pos_emb)
                    t_emb = self.get_time_emb(segment, self.iters_L - 1, iter_H)
                    for block in self.CA_H:
                        CA_mask = node_CA_mask_H
                        if self.use_im_attn_mask:
                            inc_t = self.dot_prod_incidence(q=z_H, k=z_L)
                            inc_t = torch.cat([inc_t, torch.zeros_like(inc_t).repeat(self.CA_H[0].CA.num_heads - 1, 1, 1)], dim=0) # (h*B, K, N)
                            wgt = self.im_attn_wgt_H(t_emb)
                            CA_mask += wgt*inc_t
                        z_H = block(z_H, z_L,
                                    c=t_emb,
                                    key_padding_mask_SA=edge_mask,
                                    # key_padding_mask_CA=node_mask
                                    attn_mask_CA=CA_mask
                                    )

        assert not z_H.requires_grad and not z_L.requires_grad

        ### 1-step gradient approximation
        z_L = self.norm_L(z_L + input_state)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1)
        for block in self.CA_L:
            CA_mask = node_CA_mask_L
            if self.use_im_attn_mask:
                inc_t = self.dot_prod_incidence(q=z_L, k=z_H)
                inc_t = torch.cat([inc_t, torch.zeros_like(inc_t).repeat(self.CA_L[0].CA.num_heads - 1, 1, 1)], dim=0) # (h*B, N, K)
                wgt = self.im_attn_wgt_L(t_emb)
                CA_mask += wgt*inc_t
            z_L = block(z_L, z_H,
                         c=t_emb,
                         key_padding_mask_SA=node_mask,
                        #  key_padding_mask_CA=edge_mask
                         attn_mask_CA=CA_mask
                        )

        z_H = self.norm_H(z_H + edge_pos_emb)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1)
        for block in self.CA_H:
            CA_mask = node_CA_mask_H
            if self.use_im_attn_mask:
                inc_t = self.dot_prod_incidence(q=z_H, k=z_L)
                inc_t = torch.cat([inc_t, torch.zeros_like(inc_t).repeat(self.CA_H[0].CA.num_heads - 1, 1, 1)], dim=0) # (h*B, K, N)
                wgt = self.im_attn_wgt_H(t_emb)
                CA_mask += wgt*inc_t
            z_H = block(z_H, z_L,
                         c=t_emb,
                         key_padding_mask_SA=edge_mask,
                        #  key_padding_mask_CA=node_mask
                         attn_mask_CA=CA_mask
                        )

        ### prediction
        inc = self.dot_prod_incidence(q=z_H, k=z_L) # (B, K, N)
        ind = self.indicator_predictor(z_H) # (B, K, 1)
        im = torch.cat([inc, ind], dim=2) # (B, K, N+1)

        ### new state
        state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach())

        return im, state
