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

        self.node_regression = "node_regressor" in self.config
        self.edge_regression = "edge_regressor" in self.config

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
        outputs = []
        for s in range(self.net.segments):
            out_dict = self.net(hid_state, node_feats, segment=s)
            outputs.append(out_dict)

        if return_segments:
            return outputs
        else:
            return outputs[-1]

    def edge_regression_loss(self, pred, target, indices=None, mask=None):
        '''
        Computes edge regression loss after aligning predictions with targets.
        '''
        if indices is not None:
            target_perm_idx = torch.from_numpy(indices[:,1,:]).to(pred.device).long()
            target_perm_idx_exp = target_perm_idx.unsqueeze(2).expand(-1, -1, pred.shape[2])
            target_aligned = torch.gather(target, 1, target_perm_idx_exp)
            if mask is not None:
                mask_aligned = torch.gather(mask, 1, target_perm_idx)
        else:
            target_aligned = target
            mask_aligned = mask

        if mask_aligned is not None:
            target_aligned[mask_aligned] = 0.
        normal_loss = 1 - F.cosine_similarity(pred[:,:,:3], target_aligned[:,:,:3], dim=-1)
        centroid_loss = F.mse_loss(pred[:,:,3:], target_aligned[:,:,3:], reduction='none')
        den = pred.numel()
        if mask_aligned is not None:
            normal_loss[mask_aligned] *= 0.
            centroid_loss[mask_aligned] *= 0.
            den = (~mask_aligned).sum()
        normal_loss = normal_loss.sum() / den
        centroid_loss = centroid_loss.sum() / den
        edge_regression_loss = normal_loss + centroid_loss
        return {
            "total_loss": edge_regression_loss,
            "normal_loss": normal_loss,
            "centroid_loss": centroid_loss
        }


    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            node_feats = batch['node_feats']
            im_truth = batch['incidence_matrix']
            node_targets = batch.get('node_targets', None)
            edge_targets = batch.get('edge_targets', None)
        elif len(batch) == 2:
            node_feats, im_truth = batch
        else:
            raise ValueError("Invalid batch format")

        hid_state = self.net.get_init_state()

        ### Deep Supervision
        for s in range(self.net.segments):

            self.optimizers().zero_grad()

            out_dict = self.net(hid_state, node_feats, segment=s)
            loss, indices = self.loss(out_dict["incidence_matrix"], im_truth, 
                                      n=min(self.config['nray'], node_feats.size(0)), 
                                      return_indices=True)
            loss = loss.mean()

            if self.node_regression:
                node_pred = out_dict["node_predictions"]
                node_mask = out_dict["node_mask"]
                if node_mask is not None:
                    node_targets[node_mask] = 0.
                reg_loss = F.mse_loss(node_pred, node_targets, reduction='none')
                den = node_pred.numel()
                if node_mask is not None:
                    reg_loss[node_mask] *= 0.
                    den = (~node_mask).sum()
                reg_loss = reg_loss.sum() / den
                loss = loss + reg_loss
            
            if self.edge_regression and edge_targets is not None:
                edge_pred = out_dict["edge_predictions"]
                edge_mask = im_truth[...,-1] < 0.5
                edge_reg_loss_dict = self.edge_regression_loss(edge_pred, edge_targets, indices=indices, mask=edge_mask)
                loss = loss + 0.1*edge_reg_loss_dict["total_loss"]

            self.manual_backward(loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=1.0, gradient_clip_algorithm="norm") ### TODO check if this is still helpful
            self.optimizers().step()

            hid_state = out_dict["state"].detach()

        ### Convert to probs
        probs = HHRM.preds_to_probs(out_dict["incidence_matrix"], self.net.output_norm)

        with torch.no_grad():
            logs = {
                "loss": loss,
                "mae":  metrics.mae_cardinality(probs, im_truth),
            }
        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return None
    
    def validation_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            node_feats = batch['node_feats']
            im_truth = batch['incidence_matrix']
            node_targets = batch.get('node_targets', None)
            edge_targets = batch.get('edge_targets', None)
        elif len(batch) == 2:
            node_feats, im_truth = batch
        else:
            raise ValueError("Invalid batch format")

        B, K, N = im_truth.shape

        ### Forward pass
        output_dicts = self(node_feats, return_segments=True)

        ### Align predictions
        pred_aligned, loss, indices = self.align_incidence_matrix(output_dicts[-1]['incidence_matrix'], im_truth)
        preds = [self.align_incidence_matrix(output_dict['incidence_matrix'], im_truth, indices=indices)[0] for output_dict in output_dicts]

        ### Convert to probs
        probs = [HHRM.preds_to_probs(pred, self.net.output_norm) for pred in preds]

        # d_feats = min(node_feats.shape[-1], 3) ### TODO softcode this in config.
        ### Deduce padding
        nodes_mask = torch.isnan(node_feats).any(dim=-1)
        edges_mask = im_truth[...,-1] < 0.5

        ### Note: opposite mask convention
        f1_dicts = [metrics.aligned_f1_score(im_truth, prob, 
                                             edge_mask=~edges_mask, node_mask=~nodes_mask, 
                                             threshold=0.5) for prob in probs]

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
            })

        if self.node_regression:
            node_pred = output_dicts[-1]["node_predictions"]
            if nodes_mask is not None:
                node_targets[nodes_mask] = 0.
            reg_loss = F.mse_loss(node_pred, node_targets, reduction='none')
            den = node_pred.numel()
            if nodes_mask is not None:
                reg_loss[nodes_mask] *= 0.
                den = (~nodes_mask).sum()
            reg_loss = reg_loss.sum() / den
            logs["node_reg_loss"] = reg_loss

        if self.edge_regression and edge_targets is not None:
            edge_pred = output_dicts[-1]["edge_predictions"]
            edge_reg_loss_dict = self.edge_regression_loss(edge_pred, edge_targets, indices=indices, mask=edges_mask)
            for k,v in edge_reg_loss_dict.items():
                logs[f"edge_{k}"] = v

        self.log_dict({f"{k}/val":v for k,v in logs.items()}, sync_dist=True)

        return loss
