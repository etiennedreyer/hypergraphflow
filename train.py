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
import os

def get_config(config_train, config_model=None, config_dataset=None):

    # Training config
    with open(config_train, 'r') as f:
        config = yaml.safe_load(f)

    # Config paths for model and dataset
    if config_model is not None:
        config['model'] = config_model
    elif 'model' not in config:
        raise ValueError("No model entry found in config!")
    elif not config['model'].endswith('.yaml'):
        config['model'] = f"configs/{config['model']}.yaml"

    if config_dataset is not None:
        config['dataset'] = config_dataset
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

    return config


def get_model(config):
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
    
    return model


def get_trainer(config, model_name, project_name):

    ### Logger
    logger = WandbLogger(
        name=model_name,
        project=project_name,
        log_model=False,
    )
    run = logger.experiment

    ### Log config
    config_artifact = wandb.Artifact("config", type="config")
    with open("temp_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    config_artifact.add_file("temp_config.yaml")
    run.log_artifact(config_artifact)
    os.remove("temp_config.yaml")

    ### Log code
    run.log_code(".")

    ### Checkpoints
    checkpoint_callback = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={loss/val:.4f}",
        monitor='loss/val',
        mode='min',
        save_top_k=1,
    )

    ### Training
    trainer = lightning.Trainer(
        accelerator=config.get('accelerator', 'auto'),
        devices=config.get('devices', [0]),
        max_epochs=config['num_epochs'],
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    return trainer


def train():

    ### Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_train", "-ct", type=str, required=True, help="Path to the training config file")
    parser.add_argument("--config_model", "-cm", type=str, required=False, help="Path to the model config file")
    parser.add_argument("--config_dataset", "-cd", type=str, required=False, help="Path to the dataset config file")
    args = parser.parse_args()

    ### Config
    config = get_config(args.config_train, args.config_model, args.config_dataset)

    ### Model
    model = get_model(config)

    ### Dataset
    ds_train = HyperGraphDataset(config['dataset'], config['dl_train']['total_size'])
    ds_val   = HyperGraphDataset(config['dataset'], config['dl_val']['total_size'])
    dl_train = ds_train.get_dataloader(config['dl_train'], model.name)
    dl_val   = ds_val.get_dataloader(config['dl_val'], model.name)

    ### Trainer
    trainer = get_trainer(config, model.name, ds_train.name)

    trainer.fit(model, dl_train, dl_val)

if __name__ == "__main__":
    train()