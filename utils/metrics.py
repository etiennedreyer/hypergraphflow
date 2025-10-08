import numpy as np
import ray
import torch
from torch.profiler import record_function
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from functools import partial

EPS = 1e-8

def get_lsa_indices(pdist, target_ind):

    if target_ind is None:
        indices = torch.from_numpy(np.array([linear_sum_assignment(p) for p in pdist])).long()
    else:
        b, n_part, _ = pdist.size()
        indices = torch.arange(n_part).unsqueeze(0).unsqueeze(0).expand(b, 2, -1).clone() # step1
        _arange = torch.arange(n_part)

        for i, (p, ind) in enumerate(zip(pdist, target_ind)):
            ind_mask = ind == 1
            row_ind, col_ind = linear_sum_assignment(p[ind_mask])
            col_ind = torch.from_numpy(col_ind)

            indices[i, 1, ind_mask] = col_ind # step2
            
            unmatched_mask = torch.full((n_part,), True)
            unmatched_mask[col_ind] = False
            indices[i, 1, ~ind_mask] = _arange[unmatched_mask] # step3

    return indices

@ray.remote
def lsa(arr, ind, s, e):
    ind_mask = ind[s:e] if ind is not None else None
    return get_lsa_indices(arr[s:e], ind_mask)

def ray_lsa(arr, target_ind, n):
    l = arr.shape[0]

    r = l%n
    idxs = np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

    arr_put = ray.put(arr)
    if target_ind is not None:
        ind_put = ray.put(target_ind)
    else:
        ind_put = None
    res = [lsa.remote(arr_put, ind_put, idxs[i], idxs[i+1]) for i in range(n)]
    res = torch.cat([ray.get(r) for r in res])
    return res

def kld_plus_ind_loss(input, target, ind_loss_wt=1.0, eps=1e-8, reduction='none'):

    inc_input  =  input[..., :-1]
    inc_target = target[..., :-1]
    ind_input  =  input[...,  -1:]
    ind_target = target[...,  -1:]

    inc_kld_loss = - inc_target * (inc_input + eps).log()
    ind_bce_loss = F.binary_cross_entropy(ind_input, ind_target, 
                                          reduction=reduction)

    total_loss = torch.cat(
        [inc_kld_loss, 
         ind_bce_loss * ind_loss_wt], 
         dim=-1
    )

    return total_loss

def LAP_loss(input, target, n=0, return_indices=False, masks=None, loss_fn=None, mask_nonexistent=True):

    if loss_fn is None:
        loss_fn = partial(F.binary_cross_entropy_with_logits)

    get_pdist = lambda x: loss_fn(
        x.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, x.size(1), -1),
        reduction='none'
    ).mean(3)

    pdist = get_pdist(input)
    if masks is not None:
        for mask in masks:
            pdist += get_pdist(mask)

    with record_function("LSA_computation"):

        pdist_ = pdist.detach().cpu()
        if mask_nonexistent:
            target_ind = target[..., -1].bool().cpu()
        else:
            target_ind = None

        if n > 0:
            indices = ray_lsa(pdist_, target_ind, n)
        else:
            indices = get_lsa_indices(pdist_, target_ind)

        indices = indices.to(device=pdist.device)

    flat_indices = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, flat_indices)
    total_loss = losses.mean(1)

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


