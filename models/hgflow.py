import torch
import torch.nn as nn
import yaml
from models.attention import SelfAttentionLayer, CrossAttentionLayer, DualUpdateBlock, DecoderBlock
from models.time import TimestepEmbedder
from models.mlp import MLP
import math

class HGFlow(nn.Module):

    def __init__(self, config):
        super().__init__()

        if type(config) is str:
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config

        self.flow = ('flow_match' in self.config)
        self.name = self.config['name']
        self.num_edges = self.config['num_edges']
        self.hidden_dim = self.config['hidden_dim']
        self.timestep_embedding = self.config['timestep_embedding']
        self.indicator_prediction = self.config['indicator_prediction']
        self.randomize_skip_prob = 0.0
        self.supervise_attn_mask = self.config.get('supervise_attn_mask', False)

        emb_cfg = self.config['node_embedder']
        self.node_embedder = MLP(
            input_dim=emb_cfg['input_dim'],
            layers=emb_cfg['layers'],
            output_dim=emb_cfg['output_dim'],
            activation=emb_cfg['activation']
        )

        enc_cfg = self.config['node_encoder']
        self.node_encoder_layers = nn.ModuleList([
                                    SelfAttentionLayer(
                                        model_dim=enc_cfg['model_dim'],
                                        num_heads=enc_cfg['num_heads'],
                                        activation=enc_cfg['activation'],
                                        gated=enc_cfg['gated'],
                                        ffn_factor=enc_cfg['ffn_factor'],
                                    )
                                    for _ in range(enc_cfg['num_layers'])
                                ])

        edge_mlp_cfg = self.config.get('edge_mlp', None)

        if edge_mlp_cfg is not None:
            self.edge_mlp = nn.Sequential(
                MLP(
                    input_dim=edge_mlp_cfg['input_dim'],
                    layers=edge_mlp_cfg['layers'],
                    output_dim= edge_mlp_cfg['output_dim'],
                    activation='relu'
                ),
                nn.LayerNorm(edge_mlp_cfg['output_dim']),
            )
        else:
            self.edge_mlp = None

        if self.timestep_embedding:
            self.timestep_embedder = TimestepEmbedder(self.hidden_dim)
        else:
            self.timestep_embedder = None

        ca_cfg = self.config['cross_attention']
        self.ca_layer_type = ca_cfg['layer_type']

        if self.ca_layer_type == 'dual':

            self.ca_layers = nn.ModuleList([
                DualUpdateBlock(
                    model_dim=ca_cfg['model_dim'],
                    num_heads=ca_cfg['num_heads'],
                    activation=ca_cfg['activation'],
                    c_dim=ca_cfg['model_dim'] \
                        if self.timestep_embedding else None,
                    gated=ca_cfg['gated'],
                    ffn_factor=ca_cfg['ffn_factor'],
                )
                for _ in range(ca_cfg['num_layers'])
            ])
            
        elif self.ca_layer_type == 'decoder':

            self.ca_layers = nn.ModuleList([
                DecoderBlock(
                    model_dim=ca_cfg['model_dim'],
                    num_heads=ca_cfg['num_heads'],
                    activation=ca_cfg['activation'],
                    c_dim=ca_cfg['model_dim'] if self.timestep_embedding else None,
                    gated=ca_cfg['gated'],
                    ffn_factor=ca_cfg['ffn_factor'],
                )
                for _ in range(ca_cfg['num_layers'])
            ])

        else:
            raise ValueError(f"Unknown cross attention layer type: {self.ca_layer_type}")

        if self.indicator_prediction:
            ind_pred_cfg = self.config['indicator_predictor']
            self.indicator_predictor = MLP(
                        input_dim=ind_pred_cfg['input_dim'],
                        layers=ind_pred_cfg['layers'],
                        output_dim=ind_pred_cfg['output_dim'],
                        activation=ind_pred_cfg['activation']
            )

        self.embedding = nn.Embedding(self.num_edges, self.hidden_dim)

        if self.supervise_attn_mask:
            self.n_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.h_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.sigmoid = nn.Sigmoid()

    def get_init_im(self, bs, num_edges, num_nodes, device):
        # im_0 = torch.randn(bs, num_edges, num_nodes, device=device)
        # im_0 = 0.5 + 0.1*torch.randn(bs, num_edges, num_nodes, device=device)
        im_0 = torch.rand(bs, num_edges, num_nodes, device=device)
        return im_0


    def forward(self, x, t, n):

        im_t = x

        indicator_added = True
        bs, num_edges, num_nodes = im_t.shape

        ind_t = im_t[:, :, -1:]  # indicator
        im_t  = im_t[:, :, :-1]  # incidence
        num_nodes -= 1

        ### key_padding_mask
        node_mask = torch.isnan(n).any(dim=-1)

        ### Hyperedge encoding (incidence-weighted sum of node vectors)
        if self.edge_mlp is not None:
            h = self.edge_mlp(torch.cat([
                    self.embedding.weight.unsqueeze(0).expand(bs, -1, -1),
                    torch.einsum('ben, bnf -> bef', im_t, n),
                ], dim=-1)
            )
        else:
            h = self.embedding(torch.arange(num_edges, device=im_t.device).unsqueeze(0).expand(bs, -1))

        ### Node embedding
        n = self.node_embedder(n)  # [bs, num_nodes, model_dim]

        ### Node encoding (self-attention)
        for layer in self.node_encoder_layers:
            n = layer(n, key_padding_mask=node_mask)

        ### Timestep embedding
        if t is not None:
            if len(t.shape) == 0:
                t = t.unsqueeze(0)
            t = self.timestep_embedder(t) \
                if self.timestep_embedding else None
        
        ### Node update (cross-attention)
        # q: hyperedge features
        # k/v: node features
        masks = []
        # for layer in self.decoder_layers:
        for layer in self.ca_layers:
            if self.supervise_attn_mask:
                h_proj = self.h_proj(h)
                n_proj = self.n_proj(n)
                mask = self.sigmoid((h_proj @ torch.transpose(n_proj, 1, 2)) / math.sqrt(self.hidden_dim))
                masks.append(mask)
                mask = mask.detach()
                mask = (mask < 0.1)
                zero_rows = mask.all(dim=-1)
                if zero_rows.any():
                    mask[zero_rows] = False  # ensure at least one node per hyperedge
                
                mask = mask.repeat(self.ca_layers[0].num_heads, 1, 1)  # [bs*num_heads, num_edges, num_nodes]

            else:
                # mask = None
                # mask = im_t.repeat(self.ca_layers[0].num_heads, 1, 1)  # HACK! adding im_t to attention
                mask = torch.cat([im_t, torch.zeros_like(im_t).repeat(self.ca_layers[0].num_heads - 1, 1, 1)], dim=0) # only apply to first head

            if self.ca_layer_type == 'dual':
                mask_a = torch.transpose(mask, 1, 2) if mask is not None else None

                if self.supervise_attn_mask:
                    zero_rows = mask_a.all(dim=-1)
                    if zero_rows.any():
                        mask_a[zero_rows] = False  # ensure at least one hyperedge per node

                n, h = layer(n, h, c=t, attn_mask_a=mask_a, attn_mask_b=mask)
            else:
                h = layer(h, n, c=t, attn_mask_CA=mask) #, key_padding_mask_CA=node_mask) #, key_padding_mask_SA=(ind_t.squeeze(-1) < 0.5))  # h is updated with n

        ### dot-product approach:
        inc = self.sigmoid(
                (h @ torch.transpose(n, 1, 2)) / math.sqrt(self.hidden_dim)
            )

        if self.indicator_prediction:
            ind = self.sigmoid(self.indicator_predictor(h)) # [bs, num_edges, 1]
        else:
            ind = ind_t
        
        ### Concatenate incidence and indicator predictions
        im_t = torch.cat([inc, ind], dim=2)

        for i in range(len(masks)):
            masks[i] = torch.cat([masks[i], ind], dim=2)

        if self.supervise_attn_mask and self.training:
            return im_t, masks

        return im_t