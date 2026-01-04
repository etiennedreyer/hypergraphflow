import torch

import torch.nn.functional as F
from functools import partial

from models.hhrm import HHRM
from lights.base_lightning import BaseLightning
import utils.metrics as metrics
from utils.surface import compute_surface_from_points


class HHRMLightning(BaseLightning):

    def __init__(self, model_config, train_config):
        super().__init__(model_config, train_config)

        self.net = HHRM(self.config)

        ### Need to implement deep supervision manually
        self.automatic_optimization = False

        if self.net.ultra:
            # ultra_loss_fn = partial(F.binary_cross_entropy_with_logits, pos_weight=torch.tensor(self.config.get('ultra_pos_weight', 10.0)))
            ultra_loss_fn = F.binary_cross_entropy_with_logits
            self.ultra_loss = partial(metrics.LAP_loss, loss_fn=ultra_loss_fn, hungarian=self.config.get('hungarian', True))

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

    def edge_regression_loss(self, pred, target, indices=None, mask=None, huber=False):
        '''
        Computes edge regression loss after aligning predictions with targets.
        '''
        add_normal = pred.shape[2] > 3

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
        if huber:
            centroid_loss = F.huber_loss(pred[:,:,:3], target_aligned[:,:,:3], reduction='none', delta=1.0)
        else:
            centroid_loss = F.mse_loss(pred[:,:,:3], target_aligned[:,:,:3], reduction='none')
        if add_normal:
            normal_loss = 1 - (F.cosine_similarity(pred[:,:,3:], target_aligned[:,:,3:], dim=-1)**2)
        den = pred.numel()
        if mask_aligned is not None:
            centroid_loss[mask_aligned] *= 0.
            if add_normal:
                normal_loss[mask_aligned] *= 0.
            den = (~mask_aligned).sum()
        centroid_loss = centroid_loss.sum() / den
        if add_normal:
            normal_loss = normal_loss.sum() / den
        edge_regression_loss = centroid_loss
        if add_normal:
            edge_regression_loss += normal_loss
        out_dict = {
            "total_loss": edge_regression_loss,
            "centroid_loss": centroid_loss
        }
        if add_normal:
            out_dict["normal_loss"] = normal_loss
        return out_dict


    def training_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            node_feats = batch['node_feats']
            im_truth = batch['incidence_matrix']
            edge_targets = batch.get('edge_targets', None)
            if self.net.ultra:
                ultra_im_truth = batch['vertex_polygon_incidence_matrix']
                ultra_edge_targets = batch.get('vertex_targets', None)
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

            probs = HHRM.preds_to_probs(out_dict["incidence_matrix"], self.net.output_norm)
            if "ultra_incidence_matrix" in out_dict:
                ultra_probs = HHRM.preds_to_probs(out_dict["ultra_incidence_matrix"], None)
            else:
                ultra_probs = None

            surface_dict = compute_surface_from_points(node_feats, probs, ultra_probs, eps=1e-4)

            if edge_targets is not None:
                if 'edge_predictions' in out_dict:
                    edge_pred = out_dict["edge_predictions"]
                else:
                    centroids = surface_dict['centroids']
                    normals = surface_dict['normals']
                    edge_pred = torch.cat([centroids, normals], dim=-1)
                edge_mask = im_truth[...,-1] < 0.5
                edge_reg_loss_dict = self.edge_regression_loss(edge_pred, edge_targets, indices=indices, mask=edge_mask)
                loss = loss + 0.05*edge_reg_loss_dict["total_loss"]

            if self.net.ultra:

                ### Need to rearrange the polygons according to best match on main incidence matrix
                indices = torch.from_numpy(indices[:,1]).to(im_truth.device).long()
                indices_expanded = indices.unsqueeze(1).expand(-1, ultra_im_truth.size(1), -1)
                ultra_im_truth_up_to_ind = torch.gather(ultra_im_truth[:, :, :-1], 2, indices_expanded)
                ultra_im_truth = torch.cat([ultra_im_truth_up_to_ind, ultra_im_truth[:, :, -1:]], dim=-1)

                ultra_loss, ultra_indices = self.ultra_loss(out_dict["ultra_incidence_matrix"], ultra_im_truth, 
                                                      n=min(self.config['nray'], node_feats.size(0)), 
                                                      return_indices=True)
                ultra_loss = ultra_loss.mean()
                loss = loss + ultra_loss

                # if ultra_edge_targets is not None:
                #     if 'ultraedge_predictions' in out_dict:
                #         ultra_edge_pred = out_dict["ultraedge_predictions"]
                #     else:
                #         ultra_edge_pred = surface_dict['vertices']
                        
                #     ultra_edge_mask = ultra_im_truth[...,-1] < 0.5
                #     ultra_edge_reg_loss_dict = self.edge_regression_loss(ultra_edge_pred, ultra_edge_targets, indices=ultra_indices, mask=ultra_edge_mask, huber=True)
                #     loss = loss + 0.1*ultra_edge_reg_loss_dict["total_loss"]


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

            if self.net.ultra:
                logs["ultra_loss"] = ultra_loss
                ultra_probs = HHRM.preds_to_probs(out_dict["ultra_incidence_matrix"], None)
                logs["ultra_mae"] = metrics.mae_cardinality(ultra_probs, ultra_im_truth)

        self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return None
    
    def validation_step(self, batch, batch_idx):

        if isinstance(batch, dict):
            node_feats = batch['node_feats']
            im_truth = batch['incidence_matrix']
            node_targets = batch.get('node_targets', None)
            edge_targets = batch.get('edge_targets', None)
            ultra_edge_targets = batch.get('vertex_targets', None)
        elif len(batch) == 2:
            node_feats, im_truth = batch
        else:
            raise ValueError("Invalid batch format")

        B, K, N = im_truth.shape

        ### Forward pass
        output_dicts = self(node_feats, return_segments=True)

        ### Reconstruct surface
        pred_surface_dict = compute_surface_from_points(
            points=node_feats,
            poly_point_incidence=HHRM.preds_to_probs(output_dicts[-1]['incidence_matrix'], self.net.output_norm),
            vertex_poly_incidence=HHRM.preds_to_probs(output_dicts[-1]['ultra_incidence_matrix'], None) if self.net.ultra else None,
            eps=1e-4
        )

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

        if edge_targets is not None:
            if 'edge_predictions' in output_dicts[-1]:
                edge_pred = output_dicts[-1]["edge_predictions"]
            else:
                centroids = pred_surface_dict['centroids']
                normals = pred_surface_dict['normals']
                edge_pred = torch.cat([centroids, normals], dim=-1)
            edge_reg_loss_dict = self.edge_regression_loss(edge_pred, edge_targets, indices=indices, mask=edges_mask)
            for k,v in edge_reg_loss_dict.items():
                logs[f"edge_{k}"] = v

        if self.net.ultra:

            ultra_im_truth = batch['vertex_polygon_incidence_matrix']
            indices = torch.from_numpy(indices[:,1]).to(im_truth.device).long()
            indices_expanded = indices.unsqueeze(1).expand(-1, ultra_im_truth.size(1), -1)
            ultra_im_truth_up_to_ind = torch.gather(ultra_im_truth[:, :, :-1], 2, indices_expanded)
            ultra_im_truth = torch.cat([ultra_im_truth_up_to_ind, ultra_im_truth[:, :, -1:]], dim=-1)

            ultra_loss, ultra_indices = self.ultra_loss(output_dicts[-1]["ultra_incidence_matrix"], ultra_im_truth, 
                                                  n=min(self.config['nray'], node_feats.size(0)), 
                                                  return_indices=True)
            logs["ultra_loss"] = ultra_loss.mean()

            if ultra_edge_targets is not None:
                if 'ultraedge_predictions' in output_dicts[-1]:
                    ultra_edge_pred = output_dicts[-1]["ultraedge_predictions"]
                else:
                    ultra_edge_pred = pred_surface_dict['vertices']

                ultra_edge_mask = batch['vertex_polygon_incidence_matrix'][...,-1] < 0.5
                ultra_edge_reg_loss = self.edge_regression_loss(ultra_edge_pred, ultra_edge_targets, 
                                                                indices=ultra_indices, mask=ultra_edge_mask)["total_loss"]
                logs["ultra_edge_regression_loss"] = ultra_edge_reg_loss

        self.log_dict({f"{k}/val":v for k,v in logs.items()}, sync_dist=True)

        return loss
