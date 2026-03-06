import torch
import torch.nn as nn
import torch.nn.functional as F
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
    A: torch.Tensor = None

    def detach(self):
        return HiddenState(z_L=self.z_L.detach(), z_H=self.z_H.detach(), 
                           A=self.A.detach() if self.A is not None else None)

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
        if 'persistent_A' in hrm_cfg:
            print("Using persistent A with config:", hrm_cfg['persistent_A'])
            self.persistent_A = True
            self.masked_attention_threshold = hrm_cfg['persistent_A'].get('masked_attention_threshold', None)
        else:
            self.persistent_A = False

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

    def get_init_state(self):
        return HiddenState(z_L=self.z_L_init, 
                           z_H=self.z_H_init)

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

    def dot_prod_incidence(self, q, k, key_padding_mask=None):
        out = (q @ torch.transpose(k, 1, 2)) / math.sqrt(self.hidden_dim) # (B, Nq, Nk)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(1), -50.0)
            ### Note: using 50.0 b/c -inf gives NaNs in binary_cross_entropy_with_logits
        return out

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

    def forward(self, hid_state: HiddenState, input_state: torch.Tensor, segment: int, draft=None):

        z_L = hid_state.z_L
        z_H = hid_state.z_H
        A   = hid_state.A

        if A is None and self.persistent_A:
            ### Initialize A if not passed
            shape = (input_state.shape[0], self.num_edges, input_state.shape[1])
            A = torch.zeros(shape, device=input_state.device)

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
                        for block in self.CA_L:
                            z_L = block(z_L, z_H,
                                        c=t_emb,
                                        key_padding_mask_SA=node_mask,
                                        key_padding_mask_CA=edge_mask,
                                        attn_mask_CA=A.permute(0, 2, 1) if A is not None else None
                                        )

                if not last_iter_H:
                    ### High-level update
                    z_H = self.norm_H(z_H + edge_pos_emb)
                    t_emb = self.get_time_emb(segment, self.iters_L - 1, iter_H)
                    for block in self.CA_H:
                        z_H = block(z_H, z_L,
                                    c=t_emb,
                                    key_padding_mask_SA=edge_mask,
                                    key_padding_mask_CA=node_mask,
                                    attn_mask_CA=A
                                    )
                    # ### Persistent matrix update
                    # if self.persistent_A:
                    #     A = self.dot_prod_incidence(q=z_H, k=z_L, key_padding_mask=node_mask).detach() # (B, K, N)

        assert not z_H.requires_grad and not z_L.requires_grad and (A is None or not A.requires_grad)

        ### 1-step gradient approximation
        z_L = self.norm_L(z_L + input_state)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1)
        for block in self.CA_L:
            z_L = block(z_L, z_H,
                         c=t_emb,
                         key_padding_mask_SA=node_mask,
                         key_padding_mask_CA=edge_mask,
                         attn_mask_CA=A.permute(0, 2, 1) if A is not None else None
                        )

        z_H = self.norm_H(z_H + edge_pos_emb)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1)
        for block in self.CA_H:
            z_H = block(z_H, z_L,
                         c=t_emb,
                         key_padding_mask_SA=edge_mask,
                         key_padding_mask_CA=node_mask,
                         attn_mask_CA=A
                        )

        ### prediction
        inc = self.dot_prod_incidence(q=z_H, k=z_L, key_padding_mask=node_mask) # (B, K, N)
        ind = self.indicator_predictor(z_H) # (B, K, 1)
        im = torch.cat([inc, ind], dim=2) # (B, K, N+1)

        ### normalization
        im = self.normalize_output(im)

        ### persistent A update
        if self.persistent_A:
            A = inc.detach()
            if self.masked_attention_threshold is not None:
                ### convert to boolean mask
                ### TODO: implement softmax version too
                A = A.sigmoid() < self.masked_attention_threshold
        else:
            A = None

        ### new state
        state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach(), A=A)
        return im, state


class HTRM(HHRM):

    def __init__(self, config):
        super().__init__(config)

        ### Rename stuff
        self.name = 'HTRM'
        self.iters_latent = self.iters_L
        self.iters_deep = self.iters_H

        if self.config.get('share_weights', False):
            self.norm_H = self.norm_L
            self.CA_H = self.CA_L
            if self.timestep_embedding:
                self.context_projector_H = self.context_projector_L


    def get_time_emb(self, segment: int, iter_latent: int, iter_deep: int):
        if self.timestep_embedding:
            t_step = iter_latent + iter_deep * self.iters_latent + segment * self.iters_latent * self.iters_deep
            t_frac = t_step / (self.iters_latent * self.iters_deep * self.segments)
            t_frac = torch.tensor([t_frac], dtype=torch.float32, device=self.z_L_init.device)
            t_emb = self.timestep_embedder(t_frac)
            # print(f"segment: {segment}, iter_latent: {iter_latent}, iter_deep: {iter_deep}, t_step: {t_step}, t_frac: {t_frac.item():.4f}")
            return t_emb
        else:
            return None

    def forward(self, hid_state: HiddenState, input_state: torch.Tensor, segment: int, draft=None):

        z_L = hid_state.z_L
        z_H = hid_state.z_H
        A   = hid_state.A

        if A is None and self.persistent_A:
            ### Initialize A if not passed
            shape = (input_state.shape[0], self.num_edges, input_state.shape[1])
            A = torch.zeros(shape, device=input_state.device)

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

        def latent_recursion(z_L, z_H, input_state, edge_pos_emb, iter_deep, segment, A=None):

            z_H = self.norm_H(z_H + edge_pos_emb)

            for iter_latent in range(self.iters_latent):
                z_L = self.norm_L(z_L + input_state)
                t_emb = self.get_time_emb(segment, iter_latent, iter_deep)
                for block in self.CA_L:
                    z_L = block(z_L, z_H,
                                c=t_emb,
                                key_padding_mask_SA=node_mask,
                                key_padding_mask_CA=edge_mask,
                                attn_mask_CA=A.permute(0, 2, 1) if A is not None else None
                                )

            t_emb = self.get_time_emb(segment, self.iters_latent - 1, iter_deep)
            for block in self.CA_H:
                z_H = block(z_H, z_L,
                            c=t_emb,
                            key_padding_mask_SA=edge_mask,
                            key_padding_mask_CA=node_mask,
                            attn_mask_CA=A
                            )
            return z_L, z_H
        
        def deep_recursion(z_L, z_H, input_state, edge_pos_emb, segment, A=None, iter_deep=0):
            ### Recursing z_H-1 times to improve z_L and z_H (no gradient)
            with torch.no_grad():
                for iter_deep in range(self.iters_deep - 1):
                    z_L, z_H = latent_recursion(z_L, z_H, input_state, edge_pos_emb, iter_deep, segment, A=A)
            
            ### Last iteration with gradient
            z_L, z_H = latent_recursion(z_L, z_H, input_state, edge_pos_emb, self.iters_deep - 1, segment, A=A)

            return z_L, z_H

        z_L, z_H = deep_recursion(z_L, z_H, input_state, edge_pos_emb, segment, A=A)

        ### prediction
        inc = self.dot_prod_incidence(q=z_H, k=z_L, key_padding_mask=node_mask) # (B, K, N)
        ind = self.indicator_predictor(z_H) # (B, K, 1)
        im = torch.cat([inc, ind], dim=2) # (B, K, N+1)

        ### normalization
        im = self.normalize_output(im)

        ### persistent A update
        if self.persistent_A:
            A = inc.detach()
            if self.masked_attention_threshold is not None:
                ### convert to boolean mask
                ### TODO: implement softmax version too
                A = A.sigmoid() < self.masked_attention_threshold
        else:
            A = None

        ### new state
        state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach(), A=A)

        return im, state