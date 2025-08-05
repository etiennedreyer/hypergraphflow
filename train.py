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
import random

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

    return config


def get_dataset(config):

    total_size = 0
    splits = ['train', 'val', 'test']
    split_indices = {}
    for split in splits:
        if f'dl_{split}' not in config:
            continue
        split_size = config[f'dl_{split}']['total_size']
        split_indices[split] = list(range(total_size, total_size + split_size))
        total_size += split_size

    ds = HyperGraphDataset(config['dataset'], total_size)

    dls = {}
    for split, indices in split_indices.items():
        dls[split] = ds.get_dataloader(config[f'dl_{split}'], indices=indices)

    print(f"Generated dataset {ds.name} with {total_size} examples, ")
    print(f"max nodes: {ds.max_nodes}, max edges: {ds.max_edges}")
    print(f"Splits: {', '.join(split_indices.keys())} \
          with sizes: {', '.join(str(len(indices)) for indices in split_indices.values())}")

    return ds, dls


def get_model(config):
    # Lightning instance
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


def get_trainer(config, model_name, project_name, log=True):

    if log:
        ### Logger
        logger = WandbLogger(
            name=model_name,
            project=project_name,
            log_model=False,
        )
        run = logger.experiment

        ### Log config
        config_artifact = wandb.Artifact("config", type="config")
        temp_config_path = f"config_{run.id}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        config_artifact.add_file(temp_config_path, name="config.yaml")
        run.log_artifact(config_artifact)
        os.remove(temp_config_path)

        ### Log code
        run.log_code(".")

        ### Checkpoints
        checkpoint_callback = ModelCheckpoint(
            filename="epoch={epoch}-step={step}-val_loss={loss/val:.4f}",
            monitor='loss/val',
            mode='min',
            save_top_k=1,
        )
    else:
        logger = None
        checkpoint_callback = None

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


if __name__ == "__main__":

    ### Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_train", "-ct", type=str, required=True, help="Path to the training config file")
    parser.add_argument("--config_model", "-cm", type=str, required=False, help="Path to the model config file")
    parser.add_argument("--config_dataset", "-cd", type=str, required=False, help="Path to the dataset config file")
    parser.add_argument("--mode", "-m", type=str, default="train", choices=["train", "eval", "test"], help="Mode to run the script in")
    args = parser.parse_args()

    ### Config
    config = get_config(args.config_train, args.config_model, args.config_dataset)

    ### Manually add sampler for refiner
    if 'refiner' in config['model']['name']:
        for split in ['train', 'val', 'test']:
            if f'dl_{split}' in config:
                config[f'dl_{split}']['sampler'] = True

    ### Random seed
    seed = config.get('seed', 123456)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    lightning.seed_everything(seed, workers=True)

    ### Dataset
    ds, dls = get_dataset(config)

    ### Model
    # Make sure dimensions fit those of dataset
    if 'num_node_features' in config['model']:
        config['model']['num_node_features'] = ds.in_feats
    if 'num_edges' in config['model']:
        config['model']['num_edges'] = ds.max_edges
    model = get_model(config)

    ### Trainer
    trainer = get_trainer(config, model.name, ds.name, log=(args.mode == 'train'))

    if args.mode == 'train':
        ### Train
        trainer.fit(model, dls['train'], dls['val'])
    elif args.mode == 'test':
        ### Test
        trainer.test(model, dls['test'])
    else:
        raise ValueError("Unknown mode:", args.mode)
