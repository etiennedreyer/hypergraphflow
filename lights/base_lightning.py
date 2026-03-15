import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import yaml

import sys
import utils.metrics as metrics
from functools import partial

class BaseLightning(pl.LightningModule):

    def __init__(self, model_config, train_config):
        super().__init__()

        if type(model_config) is str:
            with open(model_config, 'r') as f:
                model_config = yaml.safe_load(f)

        if type(train_config) is str:
            with open(train_config, 'r') as f:
                train_config = yaml.safe_load(f)

        for k in model_config.keys():
            if k in train_config:
                raise ValueError(f"Duplicate key {k} found in model and train config")

        self.config = {**model_config, **train_config}
        self.name = self.config['name']

        if 'particle_flow' in self.config['dataset']['name']:
            self.config['output_norm'] = 'softmax'
        
        ### Dice loss coefficient
        dice_loss_coef = self.config.get('dice_loss_coef', 0)
        if dice_loss_coef > 0:
            print(f"Using dice loss with coefficient {dice_loss_coef}")

        ### Get base loss function
        if self.config.get('output_norm', None) is None:
            loss_fn = partial(F.binary_cross_entropy_with_logits, 
                              pos_weight=torch.tensor(self.config.get('pos_weight', 1.0)))
            print("Using BCE with logits loss (assumes logits output)")
        elif dice_loss_coef > 0:
            raise NotImplementedError("Dice loss not implemented for non-logit outputs")
        elif self.config['output_norm'] == 'sigmoid':
            loss_fn = F.binary_cross_entropy
            print("Using BCE loss (assumes sigmoid on output)")
        elif self.config['output_norm'] == 'softmax':
            loss_fn = metrics.kld_plus_ind_loss
            print("Using KLD incidence and BCE indicator loss (assumes softmax on output)")
        else:
            raise ValueError(f"Unknown output_norm {self.config['output_norm']}")

        self.loss = partial(metrics.LAP_loss, loss_fn=loss_fn, dice_loss_coef=dice_loss_coef)


    def configure_optimizers(self):

        if 'refiner' in self.name:
            parameters = filter(lambda p: p.requires_grad, self.parameters())
        else:
            parameters = self.net.parameters()

        if self.config.get('optimizer', 'adam') == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.config['learning_rate'])
        elif self.config.get('optimizer') == 'adam_atan2':
            from adam_atan2_pytorch import AdamAtan2
            optimizer = AdamAtan2(parameters, lr=self.config['learning_rate'])

        scheduler_cfg = self.config.get('scheduler', None)
        if scheduler_cfg is None:
            return optimizer
        
        if scheduler_cfg['name'] == 'cosine_warmup':
            from lights.schedulers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_cfg['warmup_epochs'],
                num_training_steps=self.trainer.max_epochs
            )
        else:
            raise NotImplementedError(f"Scheduler {scheduler_cfg['name']} not implemented")

        return {'optimizer': optimizer,
                'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }

    def on_validation_epoch_end(self):

        if self.automatic_optimization:
            return
        
        sched = self.lr_schedulers()
        if sched is None:
            return

        if not self.trainer.sanity_checking:
            sched.step()