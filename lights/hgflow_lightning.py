import yaml
import torch
import pytorch_lightning as pl
from models.hgflow import HGFlow
from utils.sampler import euler_sampler
import functools

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
        return self.model(self.context, im_t, t)


from flow_matching.utils import ModelWrapper

class VelocityModelFromX1Model(ModelWrapper):

    def __init__(self, x_1_model, path):
        super().__init__(model=x_1_model)
        self.path = path

    def forward(self, x_t, t, condition=None):
        x_1_pred = super().forward(x_t, t, n=condition)
        return self.path.target_to_velocity(
            x_1=x_1_pred,
            x_t=x_t,
            t=t,
        )
        

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
            self.sampler = euler_sampler
            # self.loss = flow_loss
            self.loss = functools.partial(
                metrics.LAP_loss,
                loss_fn=torch.nn.functional.mse_loss,
            )
        else:
            self.net = HGFlow(model_config, flow=False)
            self.FM = None
            self.loss = metrics.LAP_loss

    def get_FM(self):

        '''
        from torchcfm import ConditionalFlowMatcher

        FM = ConditionalFlowMatcher(
            sigma=self.config['flow_match']['sigma']
            )

        return FM
        '''
        from flow_matching.path import AffineProbPath
        from flow_matching.path.scheduler import CondOTScheduler

        path = AffineProbPath(scheduler=CondOTScheduler())
        return path
    
    def sample_location_and_conditional_flow(self, im_1, im_0=None):

        ### standard normal by default
        if im_0 is None:
            im_0 = torch.randn_like(im_1, device=im_1.device)

        ### random timestep between 0 and 1
        t = torch.rand(im_1.shape[0], device=im_1.device)

        ### sample conditional path
        sample = self.FM.sample(t=t, x_0=im_0, x_1=im_1)

        return t, sample.x_t

    # def get_sampler(self):
    def sample(self, im_0, n, save_seq=False):

        '''
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
        '''



        return self.sampler(
            # self.net,
            VelocityModelFromX1Model(self.net, self.FM),
            x_0=im_0,
            condition=n,
            steps=self.config['sampler'].get('steps', 50),
            save_seq=save_seq
        )


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
            # t, im_t, u_t = self.FM.sample_location_and_conditional_flow(im_0, transform_im(im_truth, forward=True))
            t, im_t  = self.sample_location_and_conditional_flow(transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(im_t, t, n)
            # loss = self.loss(pred, u_t, mask=h_exists_mask)
            loss = self.loss(pred, im_truth) #, mask=h_exists_mask)
            # loss = self.loss(pred, u_t).mean()
            self.log("loss/train", loss)

        else:
            pred = self.net(im_0, t=None, n=n)
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
            t, im_t = self.sample_location_and_conditional_flow(transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(im_t, t, n)
            flow_loss = self.loss(pred, im_truth) #, mask=h_exists_mask)
            # flow_loss = self.loss(u_pred, u_t).mean()
            
            im_pred = self.sample(im_0, n)
            im_pred = im_pred * im_truth[:,:,-1:]

        else:
            im_pred = self.net(im_0, t=None, n=n)
        
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
    
