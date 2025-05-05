import yaml
import torch
import pytorch_lightning as pl
from models.hgflow import HGFlow

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
from metrics import LAP_loss

class HGFlowLightning(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        with open(train_config, 'r') as f:
            self.config = yaml.safe_load(f)
        self.net = HGFlow(model_config)
        
        self.loss = LAP_loss

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
        ind_truth = im_truth[:, :, -1]
        im_truth = im_truth[:, :, :-1]
        bs, num_edges, num_nodes = im_truth.shape

        im_0 = torch.randn(bs, num_edges, num_nodes, device=n.device)
        # im_0 = im_truth.clone() # HACK
        im_pred = self.net(n, im_0, t=None)

        loss = self.loss(im_pred, im_truth).mean()

        self.log('train/loss', loss, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        n, im_truth = batch
        ind_truth = im_truth[:, :, -1]
        im_truth = im_truth[:, :, :-1]
        bs, num_edges, num_nodes = im_truth.shape
        im_0 = torch.randn(bs, num_edges, num_nodes, device=n.device)
        im_pred = self.net(n, im_0, t=None)
        loss = self.loss(im_pred, im_truth).mean()
        self.log('val/loss', loss, prog_bar=True, logger=True)
        return loss
    
