import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from functools import partial

EPS = 1e-8

def l_split_ind(l, n):
    r = l%n
    return np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

@ray.remote
def lsa(arr, s, e):
    return np.array([linear_sum_assignment(p) for p in arr[s:e]])

def ray_lsa(arr, n):
    l = arr.shape[0]
    ind = l_split_ind(l, n)
    arr_id = ray.put(arr)
    res = [lsa.remote(arr_id, ind[i], ind[i+1]) for i in range(n)]
    res = np.concatenate([ray.get(r) for r in res])
    return res

def kld_plus_ind_loss(input, target, ind_loss_wt=1.0, eps=1e-8, reduction='none', log_inputs=True):

    inc_input  =  input[..., :-1]
    inc_target = target[..., :-1]
    ind_input  =  input[...,  -1:]
    ind_target = target[...,  -1:]

    if log_inputs:
        ### assumes log-softmax input
        inc_kld_loss = - inc_target * inc_input
    else:
        ### assumes regular softmax input
        inc_kld_loss = - inc_target * (inc_input + eps).log()

    ### Need to cast to float32 to avoid "unsafe to autocast" error
    with torch.autocast(device_type=input.device.type, enabled=False):
        ind_bce_loss = F.binary_cross_entropy(ind_input.float(), ind_target.float(), 
                                            reduction=reduction)

    total_loss = torch.cat(
        [inc_kld_loss, 
         ind_bce_loss * ind_loss_wt], 
         dim=-1
    )

    return total_loss

def LAP_loss(input, target, n=0, return_indices=False, masks=None, loss_fn=None, parallel=False, hungarian=True, node_mask=None):

    if loss_fn is None:
        loss_fn = partial(F.binary_cross_entropy_with_logits)

    B, K, N = input.shape
    device = input.device

    if hungarian:
        ### x has shape (B, K, N)
        get_cost_matrix_parallel = lambda x: loss_fn(
            x.unsqueeze(1).expand(-1, K, -1, -1), 
            target.unsqueeze(2).expand(-1, -1, K, -1),
            reduction='none'
        ).mean(3)

        ### x has shape (B, 1, N)
        get_cost_matrix_serial = lambda x: loss_fn(
            x.expand(-1, K, -1),
            target,
            reduction='none'
        ).mean(2)
        
        ### Compute lowest-loss permutation (no grad)
        with torch.no_grad():

            ### allocates (B, K, K, N) tensor
            if parallel:
                pdist = get_cost_matrix_parallel(input)
                if masks is not None:
                    for mask in masks:
                        pdist += get_cost_matrix_parallel(mask)
            
            ### allocates (B, K, N) tensor
            else:
                pdist = torch.zeros((B, K, K), device=device)

                for k in range(K):
                    input_k = input[:, k:k+1, :]
                    pdist[:, k, :] = get_cost_matrix_serial(input_k)
                    if masks is not None:
                        for mask in masks:
                            mask_k = mask[:, k:k+1, :]
                            pdist[:, k, :] += get_cost_matrix_serial(mask_k)

        ### (B, K, K) cost matrix
        cost_matrix = pdist.detach().cpu().numpy()

        ### Solve LSA with Hungarian algorithm
        if n > 0:
            indices = ray_lsa(cost_matrix, n)
        else:
            indices = np.array([linear_sum_assignment(p) for p in cost_matrix])

        ### Rearrange target rows for minimal loss
        target_perm_idx = torch.from_numpy(indices[:,1]).to(device).long() # (B, K)
        target_perm_idx_expanded = target_perm_idx.unsqueeze(2).expand(-1, -1, N) # (B, K, N)
        target_aligned = torch.gather(
            target, 1, target_perm_idx_expanded)
    else:
        target_aligned = target
        range_K = np.arange(K)
        indices = np.array([(range_K, range_K) for _ in range(B)])
    
    ### Compute loss with aligned target
    total_loss = loss_fn(input, target_aligned, reduction='none')
    
    ### Average over matrix dimensions, considering only valid entries
    if node_mask is not None:
        ### Note: assumes that True entries correspond to _padded_ nodes
        num_valid_nodes = (~node_mask).sum(dim=1)
        num_total_entries = num_valid_nodes * K + K  # valid_nodes * edges + indicators
        total_loss = total_loss.sum(dim=(1, 2)) / (num_total_entries.float() + 1e-8)
    else:
        total_loss = total_loss.mean(dim=(1, 2))

    if return_indices:
        return total_loss, indices

    return total_loss

def _error_count_indicator(gt_inc, pred_inc, d):

    ### reduce d if needed
    d = min(d, pred_inc.shape[2]-1)

    pred_m = pred_inc[...,-1] > 0.5
    gt_m = gt_inc[...,-1] > 0.5

    ### consider only edges which have at least d nodes above threshold
    thresh = 0.2
    pred_m = pred_m & ((pred_inc[...,:-1]>thresh).sum(2) >= d)
    gt_m = gt_m & ((gt_inc[...,:-1]>thresh).sum(2) >= d)

    pred_inc = pred_inc[...,:-1].topk(d, dim=2, sorted=False)[1].sort()[0]
    gt_inc = gt_inc[...,:-1].topk(d, dim=2, sorted=False)[1].sort()[0]

    # batch x edge_pred x edge_gt
    eq = (pred_inc.unsqueeze(2) == gt_inc.unsqueeze(1)).all(3)
    eq = eq * pred_m.unsqueeze(2) * gt_m.unsqueeze(1)
    tp = eq.any(1).sum(1)  # count unique only
    fp = (pred_m * ~eq.any(2)).sum(1)
    fn = (gt_m * ~eq.any(1)).sum(1)
    return tp, fp, fn

def _triu_mean(x):
    if len(x.shape) < 3:
        x = x.unsqueeze(0)
    return x.triu(1).sum((1,2)) * 2. / (x.size(1) * (x.size(1)-1))
    
def _error_count_adj(gt_adj, pred_adj):
    pred_adj = pred_adj.clamp(0, 1)
    tp = _triu_mean(gt_adj * pred_adj)
    fp = _triu_mean((1 - gt_adj) * pred_adj)
    fn = _triu_mean(gt_adj * (1 - pred_adj))
    return tp, fp, fn

def error_count(type, gt, pred, **kwargs):
    assert type in ["adj", "ind"]
    if type == "adj":
        tp, fp, fn = _error_count_adj(gt, pred)
    else:
        tp, fp, fn = _error_count_indicator(gt, pred, kwargs.get("d_feats", None))
    return tp, fp, fn

def precision(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    return tp / (tp + fp + EPS)

def recall(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    return tp / (tp + fn + EPS)

def f1_score(gt, pred, type="adj", **kwargs):
    tp, fp, fn = error_count(type, gt, pred, **kwargs)
    f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
    return f1

def aligned_f1_score(target, pred, edge_mask=None, node_mask=None, threshold=0.5):

    ### Assumes that target has already been aligned to pred
    ### target, pred: (B, K, N+1)
    ### edge_mask: (B, K) <- zero where edges are padded
    ### node_mask: (B, N) <- zero where nodes are padded

    ### Mask padded entries
    if node_mask is not None:
        indicator_mask = torch.ones((target.shape[0], 1), device=target.device)  # (B, 1)
        node_mask = torch.cat([node_mask, indicator_mask], dim=-1)  # (B, N+1)
        node_mask = node_mask.unsqueeze(1)  # (B, N+1) -> (B, 1, N+1)
        target = target * node_mask
        pred = pred * node_mask

    if edge_mask is not None:
        edge_mask_ = edge_mask.unsqueeze(2)  # (B, K) -> (B, K, 1)
        target = target * edge_mask_
        pred = pred * edge_mask_

    ### Convert to binary
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    
    ### For each hyperedge (dim -1) compute TP, FP, FN
    tp = (pred_bin * target_bin).sum(dim=-1)
    fp = (pred_bin * (1 - target_bin)).sum(dim=-1)
    fn = ((1 - pred_bin) * target_bin).sum(dim=-1)

    ### Precision
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # Average over VALID edges only
    if edge_mask is not None:
        precision_sum = (precision * edge_mask).sum()
        recall_sum = (recall * edge_mask).sum()
        f1_sum = (f1 * edge_mask).sum()
        N_valid_edges = edge_mask.sum()
        return {
            "precision": precision_sum / (N_valid_edges + 1e-8),
            "recall": recall_sum / (N_valid_edges + 1e-8),
            "f1": f1_sum / (N_valid_edges + 1e-8)
        }
    else:
        return {
            "precision": precision.mean(),
            "recall": recall.mean(),
            "f1": f1.mean()
        }

def delaunay_adj_metrics(targ_adj, pred_adj, k=2):
    diag_mask = torch.eye(pred_adj.shape[2]).repeat(pred_adj.shape[0], 1, 1).bool()
    pred_adj = (pred_adj > 0.5).int()
    pred_adj[diag_mask] = 0

    tp = (targ_adj * pred_adj).sum((1,2)).float()
    tn = ((1-targ_adj) * (1-pred_adj)).sum((1,2)).float()
    fp = ((1-targ_adj) * pred_adj).sum((1,2)).float()
    fn = (targ_adj * (1-pred_adj)).sum((1,2)).float()
    
    acc = ((tp+tn) / (tp+tn+fp+fn))
    prec = (tp / (tp+fp+EPS))
    rec = (tp / (tp+fn+EPS))
    fone = 2*tp / (2*tp+fp+fn+EPS)
    return acc, fone, prec, rec

def mae_cardinality(pred, target):
    card_targ = (pred[:,:,-1]>0.5).sum(1).float()
    card_pred = (target[:,:,-1]>0.5).sum(1).float()
    return F.l1_loss(card_targ, card_pred)

