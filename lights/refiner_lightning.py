from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import default_rng

import sys
from models.hypergraph_refiner import IterativeRefiner
from functools import partial

from lights.base_lightning import BaseLightning
import utils.metrics as metrics


def partitionfunc(n,k,l=1):
    '''
    n is the integer to partition, k is the length of partitions, l is the min partition element size
    Adapted from https://stackoverflow.com/questions/18503096/python-integer-partitioning-with-given-k-partitions
    '''
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield (n,)
        return
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

class IntegerPartitionSampler:
    def __init__(self, n, k, rng):
        self.partitions = np.array(list(partitionfunc(n, k, 0)))
        self.rng = rng

    def __call__(self):
        return self.rng.permutation(self.rng.choice(self.partitions))


class IRModel(BaseLightning):
    def __init__(self, model_config, train_config):
        super().__init__(model_config, train_config)

        self.net = IterativeRefiner(self.config)

        self.name = self.config['name']

        self.automatic_optimization = False

        self.sampler = IntegerPartitionSampler(
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

        pred = preds[-1]
        if self.net.output_norm is None:
            pred = F.sigmoid(pred)

        with torch.no_grad():
            logs = {
                "loss": loss_per_t[-1],
                "mae":  metrics.mae_cardinality(pred, target),
                **{f"loss_at{i}": l for i,l in enumerate(loss_per_upd)},
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return loss

    def eval_step(self, batch, batch_idx):
        inputs, target = batch
        pred = self(inputs)
        loss = self.loss(pred, target, n=min(self.config['nray'], inputs.size(0)))
        if self.net.output_norm is None:
            pred = F.sigmoid(pred)

        d_feats = min(inputs.shape[-1], 3) ### TODO softcode this in config.

        logs = {
            "loss": loss.mean(0),
            "f1": metrics.f1_score(target, pred, type="ind", d_feats=d_feats).mean(0),
            "precision": metrics.precision(target, pred, type="ind", d_feats=d_feats).mean(0),
            "recall": metrics.recall(target, pred, type="ind", d_feats=d_feats).mean(0),
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