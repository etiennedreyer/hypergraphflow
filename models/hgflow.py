import torch
import torch.nn as nn
import yaml
from models.attention import SelfAttentionLayer, CrossAttentionLayer, DualUpdateBlock
from models.time import TimestepEmbedder
from models.mlp import MLP

class HGFlow(nn.Module):

    def __init__(self, config, flow=True):
        super().__init__()

        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.flow = flow
        self.num_nodes = self.config['num_nodes']
        self.num_edges = self.config['num_edges']
        self.hidden_dim = self.config['hidden_dim']
        self.timestep_embedding = self.config['timestep_embedding']
        self.indicator_prediction = self.config['indicator_prediction']

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
                                    )
                                    for _ in range(enc_cfg['num_layers'])
                                ])

        ca_cfg = self.config['cross_attention']

        if self.timestep_embedding:
            self.timestep_embedder = TimestepEmbedder(ca_cfg['model_dim'])
        else:
            self.timestep_embedder = None

        edge_mlp_cfg = self.config['edge_mlp']
        self.edge_mlp = MLP(
            input_dim=edge_mlp_cfg['input_dim'],
            layers=edge_mlp_cfg['layers'],
            output_dim= edge_mlp_cfg['output_dim'],
            activation='relu'
        )

        # self.cross_attention_layers = nn.ModuleList([
        #                                 CrossAttentionLayer(
        #                                     model_dim=ca_cfg['model_dim'],
        #                                     num_heads=ca_cfg['num_heads'],
        #                                     activation=ca_cfg['activation'],
        #                                     c_dim=ca_cfg['model_dim'] \
        #                                         if self.timestep_embedding else None,
        #                                     gated=ca_cfg['gated'],
        #                                 )
        #                                 for _ in range(ca_cfg['num_layers'])
        #                             ])

        self.cross_attention_layers = nn.ModuleList([
                                        DualUpdateBlock(
                                            model_dim=ca_cfg['model_dim'],
                                            num_heads=ca_cfg['num_heads'],
                                            activation=ca_cfg['activation'],
                                            c_dim=ca_cfg['model_dim'] \
                                                if self.timestep_embedding else None,
                                            gated=ca_cfg['gated'],
                                        )
                                        for _ in range(ca_cfg['num_layers'])
                                    ])

        inc_pred_cfg = self.config['incidence_predictor']
        self.incidence_predictor = MLP(
                    input_dim=inc_pred_cfg['input_dim'],
                    layers=inc_pred_cfg['layers'],
                    output_dim=inc_pred_cfg['output_dim'],
                    activation=inc_pred_cfg['activation']
        )

        if self.indicator_prediction:
            ind_pred_cfg = self.config['indicator_predictor']
            self.indicator_predictor = MLP(
                        input_dim=ind_pred_cfg['input_dim'],
                        layers=ind_pred_cfg['layers'],
                        output_dim=ind_pred_cfg['output_dim'],
                        activation=ind_pred_cfg['activation']
            )

        self.sigmoid = nn.Sigmoid()

    def get_init_im(self, bs, num_edges, num_nodes, device):
        im_0 = torch.randn(bs, num_edges, num_nodes, device=device)
        return im_0


    def forward(self, x, t, n):

        im_t = x

        bs, num_edges, num_nodes = im_t.shape

        if num_nodes == self.num_nodes + 1:
            indicator_added = True
            ind_t = im_t[:, :, -1:]  # indicator
            im_t  = im_t[:, :, :-1]  # incidence
            num_nodes -= 1
        else:
            indicator_added = False
            ind_t = None

        assert num_nodes == self.num_nodes, f"num_nodes {num_nodes} != {self.num_nodes}"
        assert num_edges == self.num_edges, f"num_edges {num_edges} != {self.num_edges}"

        ### key_padding_mask
        node_mask = torch.isnan(n).any(dim=-1)

        # if self.flow:
        #     ### normalize input incidence matrix
        #     im_t = self.sigmoid(im_t)

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

        ### Hyperedge encoding (incidence-weighted sum of node vectors)
        h = self.edge_mlp(torch.einsum('ben, bnd -> bed', im_t, n))
        
        ### Node update (cross-attention)
        # q: node features
        # k/v: hyperedge features
        for layer in self.cross_attention_layers:
            n, h = layer(n, h, c=t)

        ### Incidence prediction
        inc = self.incidence_predictor(n) # [bs, num_nodes, num_edges]
        inc = inc.permute(0, 2, 1)        # [bs, num_edges, num_nodes]

        if indicator_added:
            if self.indicator_prediction:

                ### Indicator prediction
                ind = self.indicator_predictor(h) # [bs, num_edges, 1]

                ### Concatenate incidence and indicator predictions
                im_t = torch.cat([inc, ind], dim=2)
                # im_t = torch.cat([inc + im_t, ind + ind_t], dim=2)

            else:
                im_t = torch.cat([inc, ind_t], dim=2)

        if not self.flow:
            ### normalize output incidence matrix
            im_t = self.sigmoid(im_t)

        return im_t