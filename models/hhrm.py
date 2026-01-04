import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from models.attention import DecoderBlock, ContextProjector
from models.time import TimestepEmbedder
from models.mlp import MLP
from models.fourier import RandomFourierEmbedder
from dataclasses import dataclass
import math

@dataclass
class HiddenState:
    z_L: torch.Tensor
    z_H: torch.Tensor
    z_U: torch.Tensor = None

    def detach(self):
        if self.z_U is None:
            return HiddenState(z_L=self.z_L.detach(), z_H=self.z_H.detach())
        else:
            return HiddenState(z_L=self.z_L.detach(), z_H=self.z_H.detach(), z_U=self.z_U.detach())

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
        self.num_ultras = self.config.get('num_ultraedges', None)
        self.hidden_dim = self.config['hidden_dim']
        self.timestep_embedding = self.config['timestep_embedding']
        self.fourier_features = self.config['fourier_features']
        self.output_norm = self.config.get('output_norm', None)
        self.ultra = (self.num_ultras is not None)

        ### Hierarchical reasoning parameters
        hrm_cfg = self.config['hrm']
        self.iters_L = hrm_cfg['iters_L']
        self.iters_H = hrm_cfg['iters_H']
        self.iters_U = hrm_cfg.get('iters_U', 1)
        self.segments = hrm_cfg['segments']
        self.use_draft = hrm_cfg.get('use_draft', False)

        ### Initial, static hidden states
        self.register_buffer("z_L_init", torch.nn.init.trunc_normal_(torch.empty(1, self.hidden_dim)))
        self.register_buffer("z_H_init", torch.nn.init.trunc_normal_(torch.empty(1, self.hidden_dim)))

        ### Fourier feature embedding
        self.num_fourier_features = 0
        if self.fourier_features:
            self.num_fourier_features = 20 * self.num_node_features
            self.fourier_embedder = RandomFourierEmbedder(
                    input_dim=self.num_node_features,
                    output_dim=self.num_fourier_features,
                )

        ### Node feature embedding
        emb_cfg = self.config['node_embedder']
        self.node_embedder = MLP(
            input_dim=self.num_node_features + self.num_fourier_features,
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

        ### Node regressor
        if "node_regressor" in self.config:
            node_reg_cfg = self.config["node_regressor"]
            self.node_regressor = MLP(
                input_dim=node_reg_cfg['input_dim'],
                layers=node_reg_cfg['layers'],
                output_dim=node_reg_cfg['output_dim'],
                activation=node_reg_cfg['activation']
            )
        else:
            self.node_regressor = None

        ### Edge regressor
        if "edge_regressor" in self.config:
            edge_reg_cfg = self.config["edge_regressor"]
            self.edge_regressor = MLP(
                input_dim=edge_reg_cfg['input_dim'],
                layers=edge_reg_cfg['layers'],
                output_dim=edge_reg_cfg['output_dim'],
                activation=edge_reg_cfg['activation']
            )
        else:
            self.edge_regressor = None

        if self.ultra:
            ### Initial, static ultraedge hidden states
            self.register_buffer("z_U_init", torch.nn.init.trunc_normal_(torch.empty(1, self.hidden_dim)))

            ### Ultraedge positional embedding
            self.ultra_embedder = nn.Embedding(self.num_ultras, self.hidden_dim)

            ### Ultraedge injection layer norm
            self.norm_U = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

            ### Ultraedges (highest-level) updated based on hyperedges (high-level)
            CA_U_cfg = self.config['ultraedge_CA_U']
            if self.timestep_embedding:
                self.context_projector_U = ContextProjector(
                                            c_dim=CA_U_cfg['c_dim'],
                                            model_dim=CA_U_cfg['model_dim'],
                                            gated=CA_U_cfg['gated'],
                                            activation=CA_U_cfg['activation']
                                        )
            self.CA_U = nn.ModuleList([
                                DecoderBlock(
                                    model_dim=CA_U_cfg['model_dim'],
                                    num_heads=CA_U_cfg['num_heads'],
                                    activation=CA_U_cfg['activation'],
                                    c_dim=CA_U_cfg['c_dim'] if self.timestep_embedding else None,
                                    gated=CA_U_cfg['gated'],
                                    scaling=CA_U_cfg['scaling'],
                                    ffn_factor=CA_U_cfg['ffn_factor'],
                                    c_proj=self.context_projector_U if self.timestep_embedding else None,
                                )
                                for _ in range(CA_U_cfg['num_layers'])
                            ])

            ## Ultraindicator predictor
            ultra_ind_pred_cfg = self.config['ultraindicator_predictor']
            self.ultraindicator_predictor = MLP(
                        input_dim=ultra_ind_pred_cfg['input_dim'],
                        layers=ultra_ind_pred_cfg['layers'],
                        output_dim=ultra_ind_pred_cfg['output_dim'],
                        activation=ultra_ind_pred_cfg['activation']
            )

            if "ultraedge_regressor" in self.config:
                ### Ultraedge regressor
                ultraedge_reg_cfg = self.config["ultraedge_regressor"]
                self.ultraedge_regressor = MLP(
                    input_dim=ultraedge_reg_cfg['input_dim'],
                    layers=ultraedge_reg_cfg['layers'],
                    output_dim=ultraedge_reg_cfg['output_dim'],
                    activation=ultraedge_reg_cfg['activation']
                )
            else:
                self.ultraedge_regressor = None

        ### Logit offset
        self.logit_offset = nn.Parameter(torch.ones(1)*self.config.get('logit_offset', -4.0))
        if self.ultra:
            self.logit_offset_ultra = nn.Parameter(torch.ones(1)*self.config.get('logit_offset_ultra', -4.0))

    def get_init_state(self):
        if self.ultra:
            return HiddenState(z_L=self.z_L_init, z_H=self.z_H_init, z_U=self.z_U_init)
        else:
            return HiddenState(z_L=self.z_L_init, z_H=self.z_H_init)

    def get_time_emb(self, segment: int, iter_L: int, iter_H: int, iter_U: int = 0):
        if self.timestep_embedding:
            t_step = iter_L + iter_H * self.iters_L + iter_U * self.iters_L * self.iters_H + segment * self.iters_U * self.iters_L * self.iters_H * self.iters_U
            t_frac = t_step / (self.iters_L * self.iters_H * self.iters_U * self.segments)
            t_frac = torch.tensor([t_frac], dtype=torch.float32, device=self.z_L_init.device)
            t_emb = self.timestep_embedder(t_frac)
            # print(f"segment: {segment}, iter_L: {iter_L}, iter_H: {iter_H}, t_step: {t_step}, t_frac: {t_frac.item():.4f}")
            return t_emb
        else:
            return None

    def dot_prod_incidence(self, q, k):
        return (q @ torch.transpose(k, 1, 2)) / math.sqrt(self.hidden_dim) # (B, Nq, Nk)

    @staticmethod
    def normalize_output(im, output_norm):
        if output_norm == 'sigmoid':
            im = torch.sigmoid(im)
        elif output_norm == 'softmax':
            im = torch.cat([
                F.softmax(im[..., :-1], dim=1),
                torch.sigmoid(im[..., -1:])
            ], dim=-1)
        elif output_norm == 'log_softmax':
            im = torch.cat([
                F.log_softmax(im[..., :-1], dim=1),
                torch.sigmoid(im[..., -1:])
            ], dim=-1)
        elif output_norm is not None:
            raise ValueError(f"Unknown output_norm {output_norm}")
        return im
    
    @staticmethod
    def preds_to_probs(preds, output_norm):
        if output_norm is None:
            probs = torch.sigmoid(preds)
        elif output_norm in ['sigmoid', 'softmax']:
            probs = preds
        elif output_norm == 'log_softmax':
            probs = torch.cat([
                torch.exp(preds[..., :-1]), 
                preds[..., -1:]],
                dim=-1)
        else:
            raise ValueError(f"Unknown output_norm {output_norm}")
        return probs

    def forward(self, hid_state: HiddenState, input_state: torch.Tensor, segment: int,draft=None):

        z_L = hid_state.z_L
        z_H = hid_state.z_H
        if self.ultra:
            z_U = hid_state.z_U
        else:
            z_U = None

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
            if self.fourier_features:
                fourier_emb = self.fourier_embedder(input_state)
                input_state = torch.cat([input_state, fourier_emb], dim=-1)
            input_state = self.node_embedder(input_state)

        ### Hyperedge positional embedding
        edge_pos_idx = torch.arange(self.num_edges, device=input_state.device).unsqueeze(0).expand(input_state.shape[0], -1)
        edge_pos_emb = self.edge_embedder(edge_pos_idx)

        if self.ultra:
            ### Ultraedge positional embedding
            ultra_pos_idx = torch.arange(self.num_ultras, device=input_state.device).unsqueeze(0).expand(input_state.shape[0], -1)
            ultra_pos_emb = self.ultra_embedder(ultra_pos_idx)
            z_U = z_U + ultra_pos_emb

        ### Forward up to last iteration
        # with torch.no_grad():
        for iter_U in range(self.iters_U):
            last_iter_U = (iter_U == self.iters_U - 1)
            z_H = z_H + edge_pos_emb
            for iter_H in range(self.iters_H):
                last_iter_H = (iter_H == self.iters_H - 1)

                for iter_L in range(self.iters_L):
                    last_iter_L = (iter_L == self.iters_L - 1)

                    if not (last_iter_H and last_iter_L and last_iter_U):
                        ### Low-level update
                        z_L = self.norm_L(z_L + input_state)
                        t_emb = self.get_time_emb(segment, iter_L, iter_H, iter_U)
                        for block in self.CA_L:
                            z_L = block(z_L, z_H,
                                        c=t_emb,
                                        key_padding_mask_SA=node_mask,
                                        key_padding_mask_CA=edge_mask
                                        )

                if not (last_iter_H and last_iter_U):
                    ### High-level update
                    z_H = self.norm_H(z_H + edge_pos_emb)
                    t_emb = self.get_time_emb(segment, self.iters_L - 1, iter_H, iter_U)
                    for block in self.CA_H:
                        z_H = block(z_H, z_L,
                                    c=t_emb,
                                    key_padding_mask_SA=edge_mask,
                                    key_padding_mask_CA=node_mask
                                    )
                        
            if self.ultra and not last_iter_U:
                ### Ultra-level update
                z_U = self.norm_U(z_U + ultra_pos_emb)
                t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1, iter_U)
                for block in self.CA_U:
                    z_U = block(z_U, z_H,
                                c=t_emb,
                                key_padding_mask_SA=None,
                                key_padding_mask_CA=None
                                )

            # assert not z_H.requires_grad and not z_L.requires_grad

        ### 1-step gradient approximation
        z_L = self.norm_L(z_L + input_state)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1, self.iters_U - 1)
        for block in self.CA_L:
            z_L = block(z_L, z_H,
                        c=t_emb,
                        key_padding_mask_SA=node_mask,
                        key_padding_mask_CA=edge_mask
                        )

        z_H = self.norm_H(z_H + edge_pos_emb)
        t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1, self.iters_U - 1)
        for block in self.CA_H:
            z_H = block(z_H, z_L,
                        c=t_emb,
                        key_padding_mask_SA=edge_mask,
                        key_padding_mask_CA=node_mask
                        )
            
        if self.ultra:
            z_U = self.norm_U(z_U + ultra_pos_emb)
            t_emb = self.get_time_emb(segment, self.iters_L - 1, self.iters_H - 1, self.iters_U - 1)
            for block in self.CA_U:
                z_U = block(z_U, z_H,
                            c=t_emb,
                            key_padding_mask_SA=None,
                            key_padding_mask_CA=None
                            )

        ### prediction
        inc = self.dot_prod_incidence(q=z_H, k=z_L) # (B, K, N)
        inc = inc + self.logit_offset
        ind = self.indicator_predictor(z_H) # (B, K, 1)
        im = torch.cat([inc, ind], dim=2) # (B, K, N+1)

        ### normalization
        im = self.normalize_output(im, self.output_norm)

        ### new state
        if self.ultra:
            state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach(), z_U=z_U.detach())
        else:
            state = HiddenState(z_L=z_L.detach(), z_H=z_H.detach())

        out_dict = {"incidence_matrix": im, "state": state}

        if self.node_regressor is not None:
            node_predictions = self.node_regressor(z_L)
            if node_mask is not None:
                node_predictions[node_mask] *= 0.
            out_dict["node_predictions"] = node_predictions
            out_dict["node_mask"] = node_mask

        if self.edge_regressor is not None:
            edge_predictions = self.edge_regressor(z_H)
            out_dict["edge_predictions"] = edge_predictions

        if self.ultra:

            ultra_inc = self.dot_prod_incidence(q=z_U, k=z_H) # (B, M, K)
            ultra_inc = ultra_inc + self.logit_offset_ultra
            ultra_ind = self.ultraindicator_predictor(z_U) # (B, M, 1)
            ultra_im = torch.cat([ultra_inc, ultra_ind], dim=2) # (B, M, K+1)

            ultra_im = self.normalize_output(ultra_im, None) ### Assuming logits -> BCE with logits
            out_dict["ultra_incidence_matrix"] = ultra_im

            if self.ultraedge_regressor is not None:
                ultraedge_predictions = self.ultraedge_regressor(z_U)
                out_dict["ultraedge_predictions"] = ultraedge_predictions

        return out_dict
