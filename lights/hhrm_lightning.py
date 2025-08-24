import yaml
import torch
import pytorch_lightning as pl
from models.hhrm import HHRM

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
import metrics

import torch.nn.functional as F
from functools import partial


class HHRMLightning(pl.LightningModule):

    def __init__(self, model_config, train_config):
        super().__init__()
        if type(train_config) is str:
            with open(train_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = train_config
        self.net = HHRM(model_config)
        self.name = self.net.name
        self.loss = partial(metrics.LAP_loss, loss_fn=F.binary_cross_entropy_with_logits)

        ### Need to implement deep supervision manually
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config['learning_rate']
        )

        return optimizer
    
    def training_step(self, batch, batch_idx):

        node_feats, im_truth = batch

        hid_state = self.net.get_init_state()

        ### Deep Supervision
        for s in range(self.net.segments):

            self.optimizers().zero_grad()

            pred, hid_state = self.net(hid_state, node_feats)
            loss = self.loss(pred, im_truth).mean()

            self.manual_backward(loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            self.optimizers().step()

            hid_state = hid_state.detach()

        ### Convert to probs
        pred = torch.sigmoid(pred)

        with torch.no_grad():
            logs = {
                "loss": loss,
                "mae":  metrics.mae_cardinality(pred, im_truth),
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return None
    
    def validation_step(self, batch, batch_idx):

        node_feats, im_truth = batch

        hid_state = self.net.get_init_state()

        for s in range(self.net.segments):
            pred, hid_state = self.net(hid_state, node_feats)
        
        # print("Min: ", pred.min().item(), " Max: ", pred.max().item())
        loss = self.loss(pred, im_truth).mean()

        ### Convert to probs
        pred = torch.sigmoid(pred)

        logs = {
            "loss": loss,
            "f1": metrics.f1_score(im_truth, pred, type="ind", d_feats=node_feats.shape[-1]).mean(0),
            "precision": metrics.precision(im_truth, pred, type="ind", d_feats=node_feats.shape[-1]).mean(0),
            "recall": metrics.recall(im_truth, pred, type="ind", d_feats=node_feats.shape[-1]).mean(0),
            "mae": metrics.mae_cardinality(pred, im_truth)
        }

        self.log_dict({f"{k}/val":v for k,v in logs.items()})

        return loss
    
