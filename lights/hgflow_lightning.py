import yaml
import torch
import pytorch_lightning as pl
from models.hgflow import HGFlow
from utils.sampler import euler_sampler
# import functools

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
import metrics

def transform_im(im, forward=True):
    if forward:
        # return im*2.0 - 1.0
        return im #HACK
    else:
        # return (im + 1.0) / 2.0
        return im #HACK


from flow_matching.utils import ModelWrapper

class VelocityModelFromX1Model(ModelWrapper):

    def __init__(self, x_1_model, path):
        super().__init__(model=x_1_model)
        self.path = path

    def forward(self, x, t, **model_extras):

        condition = model_extras['model_extras'].get('condition', None)

        x_1_pred = self.model(x, t, n=condition)

        return self.path.target_to_velocity(
            x_1=x_1_pred,
            x_t=x,
            t=t,
        )
        

class HGFlowLightning(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        if type(train_config) is str:
            with open(train_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = train_config
        self.loss = metrics.LAP_loss
        self.net = HGFlow(model_config)
        self.name = self.net.name
        
        if self.net.flow:
            self.FM = self.get_FM(model_config['flow_match'])
            # self.sampler = euler_sampler
        else:
            self.FM = None
            self.name = 'transformer'

        if self.net.supervise_attn_mask:
            self.name += "_masked"

    def get_FM(self, config_FM):

        from flow_matching.path import AffineProbPath
        import flow_matching.path.scheduler as schedulers

        if config_FM.get('scheduler', 'ot') == 'ot':
            path = AffineProbPath(scheduler=schedulers.CondOTScheduler())
        elif config_FM['scheduler'].startswith('poly'):
            power = float(config_FM['scheduler'].split('poly')[-1])
            path = AffineProbPath(scheduler=schedulers.PolynomialConvexScheduler(n=power))
        else:
            raise NotImplementedError

        return path
    
    def sample_location_and_conditional_flow(self, im_1, im_0=None):

        ### source distribution
        if im_0 is None:
            im_0 = self.net.get_init_im(*im_1.shape, device=im_1.device) #0.2 + 0.05*torch.randn_like(im_1, device=im_1.device)

        ### align source with target with minimal loss
        im_0, _, _ = self.align_incidence_matrix(im_0, im_1)

        ### random timestep between 0 and 1
        t = torch.rand(im_1.shape[0], device=im_1.device)
        # t = torch.pow(t, 3)
        ### sample conditional path
        sample = self.FM.sample(t=t, x_0=im_0, x_1=im_1)

        return t, sample.x_t

    def sample(self, im_0, n, save_seq=False, num_steps=12):

        from flow_matching.solver import ODESolver
        solver = ODESolver(velocity_model=VelocityModelFromX1Model(self.net, self.FM))
        return solver.sample(x_init=im_0, 
                             method='euler',
                             step_size=1.0 / num_steps,
                             time_grid=torch.linspace(0, 1, num_steps),
                             model_extras={'condition': n},
                             return_intermediates=save_seq
                            )

        # return self.sampler(
        #     VelocityModelFromX1Model(self.net, self.FM),
        #     x_0=im_0,
        #     condition=n,
        #     steps=self.config['sampler'].get('steps', 25),
        #     save_seq=save_seq
        # )

    def align_incidence_matrix(self, im_pred, im_true):

        im_pred = torch.clamp(im_pred, 0, 1)
        loss, indices = self.loss(im_pred, im_true, return_indices=True)
        indices = torch.from_numpy(indices[:,1,...]).to(im_pred.device).long()
        indices = indices.unsqueeze(-1).expand(-1, -1, im_pred.shape[2])

        im_pred_aligned = torch.gather(im_pred, 1, indices)

        return im_pred_aligned, loss, indices

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
            t, im_t  = self.sample_location_and_conditional_flow(transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(im_t, t, n)
            masks = None
            if self.net.supervise_attn_mask:
                pred, masks = pred
            
            loss = self.loss(pred, im_truth, masks=masks).mean()

            self.log("loss/train", loss)

        else:
            pred = self.net(im_0, t=None, n=n)
            masks = None
            if self.net.supervise_attn_mask:
                pred, masks = pred

            loss = self.loss(pred, im_truth, masks=masks).mean()

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
            t, im_t = self.sample_location_and_conditional_flow(transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(im_t, t, n)
            flow_loss = self.loss(pred, im_truth).mean()
            
            im_pred = self.sample(im_0, n)
            im_pred = transform_im(im_pred, forward=False)
            im_pred = torch.clamp(im_pred, 0, 1)
            # im_pred = im_pred * im_truth[:,:,-1:]

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
    
