import ray
import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils.dataset import HyperGraphDataset
import wandb
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_train", "-ct", type=str, required=True, help="Path to the training config file")
parser.add_argument("--config_model", "-cm", type=str, required=False, help="Path to the model config file")
parser.add_argument("--config_dataset", "-cd", type=str, required=False, help="Path to the dataset config file")
args = parser.parse_args()

# Training config
with open(args.config_train, 'r') as f:
    config = yaml.safe_load(f)

# Config paths for model and dataset
if args.config_model is not None:
    config['model'] = args.config_model
elif 'model' not in config:
    raise ValueError("No model entry found in config!")
elif not config['model'].endswith('.yaml'):
    config['model'] = f"configs/{config['model']}.yaml"

if args.config_dataset is not None:
    config['dataset'] = args.config_dataset
elif 'dataset' not in config:
    raise ValueError("No dataset entry found in config!")
elif not config['dataset'].endswith('.yaml'):
    config['dataset'] = f"configs/{config['dataset']}.yaml"

with open(config['model'], 'r') as f:
    config['model'] = yaml.safe_load(f)

with open(config['dataset'], 'r') as f:
    config['dataset'] = yaml.safe_load(f)

# Make sure num input features matches dataset
if 'num_node_features' in config['model']:
    config['model']['num_node_features'] = config['dataset']['D']
### TODO: do the same for num_edges

# Lightning instance
lightning.seed_everything(config.get('seed', 123456))
num_ray = config.get('nray', 0)
if num_ray > 0:
    ray.init(num_cpus=num_ray, include_dashboard=False)

if 'hgflow' in config['model']['name']:
    from lights.hgflow_lightning import HGFlowLightning

    model = HGFlowLightning(
        model_config=config['model'],
        train_config=config,
    )
elif 'refiner' in config['model']['name']:
    from lights.refiner_lightning import IRModel

    model = IRModel(
        model_config=config['model'],
        train_config=config,
    )
else:
    raise ValueError("Unknown model type:", config['model']['name'])

# Dataset loading
ds_train = HyperGraphDataset(config['dataset'], config['dl_train']['total_size'])
ds_val   = HyperGraphDataset(config['dataset'], config['dl_val']['total_size'])
dl_train = ds_train.get_dataloader(config['dl_train'], model.name)
dl_val   = ds_val.get_dataloader(config['dl_val'], model.name)

logger = WandbLogger(
    name=model.name,
    project=ds_train.name,
    log_model=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor='loss/val',
    mode='min',
)

trainer = lightning.Trainer(
    accelerator=config.get('accelerator', 'auto'),
    devices=config.get('devices', [0]),
    max_epochs=config['num_epochs'],
    check_val_every_n_epoch=1,
    logger=logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, dl_train, dl_val)