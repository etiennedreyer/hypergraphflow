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

    def align_incidence_matrix(self, im_pred, im_true):

        loss, indices = self.loss(im_pred, im_true, return_indices=True)
        indices = torch.from_numpy(indices[:,1,...]).to(im_pred.device).long()
        indices = indices.unsqueeze(-1).expand(-1, -1, im_pred.shape[2])

        im_pred_aligned = torch.gather(im_pred, 1, indices)

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
        B, K, N = im_truth.shape

        # Detect node padding mask (True = padded)
        node_mask = torch.isnan(node_feats).any(dim=-1)

        hid_state = self.net.get_init_state()

        target_indices = None
        do_hungarian = True
        ### Deep Supervision
        for s in range(self.net.segments):

            self.optimizers().zero_grad()

            if self.net.persistent_A and self.net.masked_attention_frequency == 'layer':
                pred, hid_state, A_per_layer = self.net(hid_state, node_feats, segment=s, return_A_per_layer=True)
                ### HACK: manually append indicator to masks
                ind = pred[:, :, -1:]
                A_per_layer = [torch.cat([mask, ind], dim=2) for mask in A_per_layer]
            else:
                pred, hid_state = self.net(hid_state, node_feats, segment=s, return_A_per_layer=False)
                A_per_layer = None

            # ### Rearrange ground truth based on 0th segment prediction; switch off hungarian for later segments
            # if s == 1:
            #     target_perm_idx = torch.from_numpy(target_indices[:,1]).to(im_truth.device).long() 
            #     target_perm_idx_expanded = target_perm_idx.unsqueeze(2).expand(-1, -1, N)
            #     im_truth = torch.gather(im_truth, 1, target_perm_idx_expanded)
            #     do_hungarian = False

            loss, target_indices = self.loss(pred, im_truth, 
                                      n=min(self.config['nray'], B), 
                                      return_indices=True,
                                      parallel=False,
                                      hungarian=do_hungarian,
                                      node_mask=node_mask,
                                      masks=A_per_layer)
            loss = loss.mean()

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

        # Detect node padding mask (True = padded)
        node_mask = torch.isnan(node_feats).any(dim=-1)

        preds = self(node_feats, return_segments=True)
        loss = self.loss(preds[-1], im_truth, 
                        n=min(self.config['nray'], node_feats.size(0)),
                        node_mask=node_mask).mean()

        ### Convert to probs
        probs = [HHRM.preds_to_probs(pred, self.net.output_norm) for pred in preds]

        d_feats = min(node_feats.shape[-1], 3) ### TODO softcode this in config.

        logs = {
            "loss": loss,
            "f1": metrics.f1_score(im_truth, probs[-1], type="ind", d_feats=d_feats).mean(0),
            "precision": metrics.precision(im_truth, probs[-1], type="ind", d_feats=d_feats).mean(0),
            "recall": metrics.recall(im_truth, probs[-1], type="ind", d_feats=d_feats).mean(0),
            "mae": metrics.mae_cardinality(probs[-1], im_truth),
            # "logit_mean": preds[-1].mean(),
            # "logit_std": preds[-1].std(),
            # "logit_min": preds[-1].min(),
            # "logit_max": preds[-1].max(),
        }

        for i, prob in enumerate(probs[:-1]):
            logs.update({
                f"f1_s{i}": metrics.f1_score(im_truth, prob, type="ind", d_feats=d_feats).mean(0),
            })

        self.log_dict({f"{k}/val":v for k,v in logs.items()})

        return loss


from models.hhrm import HTRM
class HTRMLightning(HHRMLightning):

    def __init__(self, model_config, train_config):
        super().__init__(model_config, train_config)

        self.net = HTRM(self.config)

        ### Need to implement deep supervision manually
        self.automatic_optimization = False