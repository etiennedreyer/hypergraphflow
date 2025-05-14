import yaml
import torch
import pytorch_lightning as pl
from models.hgflow import HGFlow

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
import metrics

def transform_im(im, forward=True):
    if forward:
        return im*2.0 - 1.0
    else:
        return (im + 1.0) / 2.0

def flow_loss(u_pred, u_true, mask=None):

    bs = u_pred.shape[0]

    loss_inc = torch.nn.functional.mse_loss(
        u_pred[:,:,:-1], u_true[:,:,:-1],
        reduction='none'
    )

    if mask is not None:
        loss_inc[~mask] = 0.0

    loss_inc = loss_inc.view(bs, -1).mean(dim=-1).mean()

    loss_ind = torch.nn.functional.mse_loss(
        u_pred[:,:,-1], u_true[:,:,-1], 
        reduction='none'
    )

    loss_ind = loss_ind.view(bs, -1).mean(dim=-1).mean()

    return loss_inc + loss_ind


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, context):
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, t, im_t, args):
        return self.model(self.context, im_t[:,:,:-1], t)


class HGFlowLightning(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        with open(train_config, 'r') as f:
            self.config = yaml.safe_load(f)
        self.name = self.config['name']
        
        if 'flow_match' in self.config:
            self.net = HGFlow(model_config, flow=True)
            self.FM = self.get_FM()
            # self.sampler = self.get_sampler()
            self.loss = flow_loss
        else:
            self.net = HGFlow(model_config, flow=False)
            self.FM = None
            self.loss = metrics.LAP_loss

    def get_FM(self):

        from torchcfm import ConditionalFlowMatcher

        FM = ConditionalFlowMatcher(
            sigma=self.config['flow_match']['sigma']
            )

        return FM        

    # def get_sampler(self):
    def sample(self, n, im_0):

        from torchdyn.core import NeuralODE

        node = NeuralODE(
                torch_wrapper(self.net, context=n),
                solver=self.config['sampler'].get('solver', 'dopri5'),
                sensitivity=self.config['sampler'].get('sensitivity', 'adjoint'),
                atol=self.config['sampler'].get('atol', 1e-4),
                rtol=self.config['sampler'].get('rtol', 1e-4),
            )

        # return node

        with torch.no_grad():
            
            ### note, passing context aside from x and t
            ### is not supported in torchdyn yet (!)
            # self.sampler.vf.set_context(n)

            seq = node(
                x=im_0,
                t_span=torch.linspace(0, 1, 25),
                save_at=[1],
            )[1]

            seq = transform_im(seq, forward=False)

        return seq[-1], seq

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config['learning_rate']
        )

        return optimizer
    
    def training_step(self, batch, batch_idx):

        n, im_truth = batch
        bs, num_edges, num_nodes = im_truth.shape
        im_0 = self.net.get_init_im(bs, num_edges, num_nodes, n.device)

        if self.FM is not None:
            h_exists_mask = im_truth[:,:,-1] > 0
            t, im_t, u_t = self.FM.sample_location_and_conditional_flow(im_0, transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(n, im_t[:,:,:-1], t=t)
            loss = self.loss(pred, u_t, mask=h_exists_mask)
            self.log("loss/train", loss)

        else:
            pred = self.net(n, im_0[:,:,:-1], t=None)
            loss = self.loss(pred, im_truth).mean()

            with torch.no_grad():
                logs = {
                    "loss": loss,
                    "mae":  metrics.mae_cardinality(pred, im_truth),
                }
            self.log_dict({f"{k}/train":v for k,v in logs.items()})

        return loss
    
    def validation_step(self, batch, batch_idx):

        n, im_truth = batch
        bs, num_edges, num_nodes = im_truth.shape
        im_0 = self.net.get_init_im(bs, num_edges, num_nodes, n.device)

        if self.FM is not None:
            h_exists_mask = im_truth[:,:,-1] > 0
            t, im_t, u_t = self.FM.sample_location_and_conditional_flow(im_0, transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            u_pred = self.net(n, im_t[:,:,:-1], t=t)
            flow_loss = self.loss(u_pred, u_t, mask=h_exists_mask)
            
            im_pred = self.sample(n, im_0)[0]

        else:
            im_pred = self.net(n, im_0[:,:,:-1], t=None)
        
        loss = metrics.LAP_loss(im_pred, im_truth).mean()

        logs = {
            "loss": loss,
            "f1": metrics.f1_score(im_truth, im_pred, type="ind", d_feats=n.shape[-1]).mean(0),
            "precision": metrics.precision(im_truth, im_pred, type="ind", d_feats=n.shape[-1]).mean(0),
            "recall": metrics.recall(im_truth, im_pred, type="ind", d_feats=n.shape[-1]).mean(0),
            "mae": metrics.mae_cardinality(im_pred, im_truth)
        }

        if self.FM is not None:
            logs["flow"] = flow_loss

        self.log_dict({f"{k}/val":v for k,v in logs.items()})

        return loss
    
