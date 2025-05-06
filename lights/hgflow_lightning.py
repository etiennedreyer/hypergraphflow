import yaml
import torch
import pytorch_lightning as pl
from models.hgflow import HGFlow
# from torchcfm import ConditionalFlowMatcher

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
import metrics

class HGFlowLightning(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        with open(train_config, 'r') as f:
            self.config = yaml.safe_load(f)
        self.net = HGFlow(model_config)
        
        self.loss = metrics.LAP_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config['learning_rate']
        )

        return optimizer

    def forward(self, n, im_t, t):
        return self.net(n, im_t, t)
    
    def training_step(self, batch, batch_idx):

        n, im_truth = batch
        bs, num_edges, num_nodes = im_truth.shape
        num_nodes -= 1 # remove indicator node

        im_0 = self.net.get_init_im(bs, num_edges, num_nodes, n.device)
        im_pred = self.net(n, im_0, t=None)

        loss = self.loss(im_pred, im_truth).mean()

        with torch.no_grad():
            logs = {
                "loss": loss,
                "mae":  metrics.mae_cardinality(im_pred, im_truth),
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return loss
    
    def validation_step(self, batch, batch_idx):

        n, im_truth = batch
        bs, num_edges, num_nodes = im_truth.shape
        num_nodes -= 1 # remove indicator node
        im_0 = self.net.get_init_im(bs, num_edges, num_nodes, n.device)
        im_pred = self.net(n, im_0, t=None)
        loss = self.loss(im_pred, im_truth).mean()

        logs = {
            "loss": loss,
            "f1": metrics.f1_score(im_truth, im_pred, type="ind", d_feats=n.shape[-1]).mean(0),
            "precision": metrics.precision(im_truth, im_pred, type="ind", d_feats=n.shape[-1]).mean(0),
            "recall": metrics.recall(im_truth, im_pred, type="ind", d_feats=n.shape[-1]).mean(0),
            "mae": metrics.mae_cardinality(im_pred, im_truth)
        }

        self.log_dict({f"{k}/val":v for k,v in logs.items()})

        return loss
    
