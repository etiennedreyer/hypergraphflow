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

class Velociwrapper(ModelWrapper):

    def __init__(self, model, path=None, is_x1_model=False):
        super().__init__(model=model)
        self.path = path
        self.is_x1_model = is_x1_model
        if is_x1_model:
            assert path is not None, "path needed to wrap an x1 model"

    def forward(self, x, t, condition): # **model_extras):

        # condition = model_extras['model_extras'].get('condition', None)

        pred = self.model(x, t, n=condition)

        # if self.model.predict_velocity:
        #     # HACK: transform to match the mean and std of the target
        #     mean_u_t = -0.4200165271759033
        #     std_u_t = 0.26402512192726135
        #     pred = pred*std_u_t + mean_u_t

        # if self.is_x1_model:
        #     pred = self.path.target_to_velocity(
        #         x_1=pred,
        #         x_t=x,
        #         t=t,
        #     )
        
        return pred

class HGFlowLightning(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        if type(train_config) is str:
            with open(train_config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = train_config
        self.net = HGFlow(model_config)
        self.name = self.net.name
        self.loss = metrics.LAP_loss
        
        if self.net.flow:
            if self.net.predict_velocity:
                self.loss = torch.nn.MSELoss()
            self.FM = self.get_FM(model_config['flow_match'])
            self.sampler = euler_sampler
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
    
    def sample_location_and_conditional_flow(self, im_1, im_0=None, t=None, node_mask=None):

        ### source distribution
        if im_0 is None:
            im_0 = self.net.get_init_im(*im_1.shape, device=im_1.device, node_mask=node_mask) #0.2 + 0.05*torch.randn_like(im_1, device=im_1.device)

        ### align source with target with minimal loss
        # im_0, _, _ = self.align_incidence_matrix(im_0, im_1)

        ### random timestep between 0 and 1
        if t is None:
            t = torch.rand(im_1.shape[0], device=im_1.device)
            # t = torch.pow(t, 3)
        ### sample conditional path
        sample = self.FM.sample(t=t, x_0=im_0, x_1=im_1)

        u = sample.dx_t
        # if self.net.predict_velocity:
        #     # HACK: transform to match the mean and std of the target
        #     mean_u_t = -0.4200165271759033
        #     std_u_t = 0.26402512192726135
        #     u = (u - mean_u_t) / std_u_t

        return t, sample.x_t, sample.dx_t

    def sample(self, im_0, n, save_seq=False, num_steps=24):

        # from flow_matching.solver import ODESolver
        # velocity_model = Velociwrapper(self.net, self.FM, is_x1_model=(not self.net.predict_velocity))
        # solver = ODESolver(velocity_model=velocity_model)
        # return solver.sample(x_init=im_0,
        #                      method='euler',
        #                      step_size=1.0 / num_steps,
        #                      time_grid=torch.linspace(0, 1, num_steps),
        #                      model_extras={'condition': n},
        #                      return_intermediates=save_seq
        #                     )

        return euler_sampler(
            Velociwrapper(self.net, self.FM, is_x1_model=(not self.net.predict_velocity)),
            x_0=im_0,
            condition=n,
            steps=num_steps,
            save_seq=save_seq
        )

    def align_incidence_matrix(self, im_pred, im_true):

        im_pred = torch.clamp(im_pred, 0, 1)
        loss, indices = metrics.LAP_loss(im_pred, im_true, return_indices=True)
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
        node_mask = torch.isnan(n).any(dim=-1)
        im_0 = self.net.get_init_im(bs, num_edges, num_nodes, n.device, node_mask=node_mask)

        if self.FM is not None:
            t, im_t, u_t  = self.sample_location_and_conditional_flow(transform_im(im_truth, forward=True), node_mask=node_mask)
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(im_t, t, n)
            masks = None
            if self.net.supervise_attn_mask:
                pred, masks = pred
            
            if self.net.predict_velocity:
                target = u_t
            else:
                target = im_truth

            if not self.net.indicator_prediction:
                pred = torch.cat([pred[:,:,:-1], target[:,:,-1:]], dim=-1)

            if self.net.supervise_attn_mask:
                loss = self.loss(pred, target, masks=masks).mean()
            else:
                loss = self.loss(pred, target).mean()

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
            t, im_t, u_t = self.sample_location_and_conditional_flow(transform_im(im_truth, forward=True))
            im_t = torch.clamp(transform_im(im_t, forward=False), 0, 1)
            pred = self.net(im_t, t, n)
            if self.net.predict_velocity:
                target = u_t
            else:
                target = im_truth
            if not self.net.indicator_prediction:
                pred = torch.cat([pred[:,:,:-1], target[:,:,-1:]], dim=-1)
            flow_loss = self.loss(pred, target).mean()

            im_pred = self.sample(im_0, n)
            im_pred = transform_im(im_pred, forward=False)
            im_pred = torch.clamp(im_pred, 0, 1)
            if not self.net.indicator_prediction:
                im_pred = torch.cat([im_pred[:,:,:-1], im_truth[:,:,-1:]], dim=-1)
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
    
