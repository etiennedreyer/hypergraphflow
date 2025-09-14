import torch
import torch.nn.functional as F
from numpy.random import default_rng

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
from hypergraph_refiner import IterativeRefiner
import metrics
import misc

from lights.base_lightning import BaseLightning


class IRModel(BaseLightning):
    def __init__(self, model_config, train_config):
        super().__init__(model_config, train_config)

        self.net = IterativeRefiner(
                    self.config['num_edges'],
                    self.config['num_node_features'],
                    self.config['hidden_dim'],
                    self.config['iters_total'],
                    return_logits=True,
            )

        self.name = self.config['name']

        self.automatic_optimization = False

        self.sampler = misc.IntegerPartitionSampler(
            self.config['iters_total']-self.config['iters_bptt']*self.config['blocks_bptt'],
            self.config['blocks_bptt'],
            default_rng(self.config['seed'])
        )

    def forward(self, inputs):
        e_t, v_t, i_t = self.net.get_initial(inputs)
        pred = self.net(inputs, e_t, v_t, i_t, t_skip=self.config['iters_total']-1, t_bp=1)[0][-1]
        return pred

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        bs = inputs.size(0)

        opt = self.optimizers()
        opt.zero_grad()
        loss_per_upd = []
        e_t, v_t, i_t = self.net.get_initial(inputs)
        t_pre = self.sampler()

        for t in t_pre:
            preds, e_t, v_t, i_t = self.net(inputs, e_t, v_t, i_t, t_skip=t, t_bp=self.config['iters_bptt'])
            loss_per_t = [self.loss(p, target, n=min(self.config['nray'], bs)).mean(0) for p in preds]
            loss = sum(loss_per_t) / self.config['iters_bptt']

            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
            e_t, v_t, i_t = e_t.detach(), v_t.detach(), i_t.detach()
            loss_per_upd.append(loss.detach())

        with torch.no_grad():
            logs = {
                "loss": loss_per_t[-1],
                "mae":  metrics.mae_cardinality(F.sigmoid(preds[-1]), target),
                **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)},
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return loss

    def eval_step(self, batch, batch_idx):
        inputs, target = batch
        pred = self(inputs)
        loss = self.loss(pred, target, n=min(self.config['nray'], inputs.size(0)))
        pred = F.sigmoid(pred)
        logs = {
            "loss": loss.mean(0),
            "f1": metrics.f1_score(target, pred, type="ind", d_feats=self.config['num_node_features']).mean(0),
            "precision": metrics.precision(target, pred, type="ind", d_feats=self.config['num_node_features']).mean(0),
            "recall": metrics.recall(target, pred, type="ind", d_feats=self.config['num_node_features']).mean(0),
            "mae": metrics.mae_cardinality(pred, target)
        }
        return logs

    def validation_step(self, batch, batch_idx):
        logs = self.eval_step(batch, batch_idx)
        self.log_dict({f"{k}/val":v for k,v in logs.items()})
        return logs["loss"]

    def test_step(self, batch, batch_idx):
        logs = self.eval_step(batch, batch_idx)
        self.log_dict({f"{k}/test":v for k,v in logs.items()})
        return logs["loss"]