import torch

import torch.nn.functional as F
from functools import partial

from models.hhrm import HHRM
from lights.base_lightning import BaseLightning
import utils.metrics as metrics


class HHRMLightning(BaseLightning):

    def __init__(self, model_config, train_config):
        super().__init__(model_config, train_config)

        self.net = HHRM(self.config)

        ### Need to implement deep supervision manually
        self.automatic_optimization = False

    def align_incidence_matrix(self, im_pred, im_true, indices=None):

        '''
        This aligns the prediction with the target, for visualization purposes.
        '''
        if indices is None:
            loss, indices = self.loss(im_pred, im_true, return_indices=True)
        else:
            loss = None
        target_perm_idx = torch.from_numpy(indices[:,1,:]).to(im_pred.device).long()
        pred_perm_idx = torch.argsort(target_perm_idx, dim=1)
        pred_perm_idx = pred_perm_idx.unsqueeze(2).expand(-1, -1, im_pred.shape[2])

        im_pred_aligned = torch.gather(im_pred, 1, pred_perm_idx)

        return im_pred_aligned, loss, indices

    def forward(self, node_feats, return_segments=False):

        hid_state = self.net.get_init_state()
        preds = []
        for s in range(self.net.segments):
            pred, hid_state = self.net(hid_state, node_feats, segment=s)
            preds.append(pred)

        if return_segments:
            return preds
        else:
            return preds[-1]

    def training_step(self, batch, batch_idx):

        node_feats, im_truth = batch

        hid_state = self.net.get_init_state()

        ### Deep Supervision
        for s in range(self.net.segments):

            self.optimizers().zero_grad()

            pred, hid_state = self.net(hid_state, node_feats, segment=s)
            loss = self.loss(pred, im_truth, n=min(self.config['nray'], node_feats.size(0))).mean()

            self.manual_backward(loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=1.0, gradient_clip_algorithm="norm") ### TODO check if this is still helpful
            self.optimizers().step()

            hid_state = hid_state.detach()

        ### Convert to probs
        probs = HHRM.preds_to_probs(pred, self.net.output_norm)

        with torch.no_grad():
            logs = {
                "loss": loss,
                "mae":  metrics.mae_cardinality(probs, im_truth),
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return None
    
    def validation_step(self, batch, batch_idx):

        node_feats, im_truth = batch

        B, K, N = im_truth.shape

        ### Forward pass
        preds = self(node_feats, return_segments=True)

        ### Align predictions
        pred_aligned, loss, indices = self.align_incidence_matrix(preds[-1], im_truth)
        preds = [self.align_incidence_matrix(pred, im_truth, indices=indices)[0] for pred in preds]

        ### Convert to probs
        probs = [HHRM.preds_to_probs(pred, self.net.output_norm) for pred in preds]

        # d_feats = min(node_feats.shape[-1], 3) ### TODO softcode this in config.
        ### Deduce padding
        nodes_mask = ~(torch.isnan(node_feats).any(dim=-1))
        edges_mask = im_truth[...,-1] > 0.5

        f1_dicts = [metrics.aligned_f1_score(im_truth, prob, edge_mask=edges_mask, node_mask=nodes_mask, threshold=0.5) for prob in probs]

        logs = {
            "loss": loss.mean(),
            "f1": f1_dicts[-1]["f1"],
            "precision": f1_dicts[-1]["precision"],
            "recall": f1_dicts[-1]["recall"],
            "mae": metrics.mae_cardinality(probs[-1], im_truth),
            "logit_mean": probs[-1].mean(),
            "logit_std": probs[-1].std(),
            "logit_min": probs[-1].min(),
            "logit_max": probs[-1].max(),
        }

        for i in range(len(probs)-1):
            logs.update({
                f"f1_s{i}": f1_dicts[i]["f1"],
                # f"f1_s{i}": metrics.f1_score(im_truth, prob, type="ind", d_feats=d_feats).mean(0),
            })

        self.log_dict({f"{k}/val":v for k,v in logs.items()}, sync_dist=True)

        return loss
