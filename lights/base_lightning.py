import torch
import pytorch_lightning as pl
import yaml

import sys
sys.path.append("../../recurrently_predicting_hypergraphs/")
import metrics

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
        self.loss = metrics.LAP_loss

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