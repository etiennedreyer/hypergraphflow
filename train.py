from pathlib import Path

import ray
import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

import sys
sys.path.append("../recurrently_predicting_hypergraphs/")

from convex_hull_dataset import get_ch_dl, ConvexHullData

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_train", "-ct", type=str, required=True, help="Path to the training config file")
parser.add_argument("--config_model", "-cm", type=str, required=True, help="Path to the model config file")
args = parser.parse_args()

# Dataset
D_FEATS = 3
N_POINTS = torch.arange(30,31)
UNIT_NORM = False
ADD_INDICATOR = True

# Model hyperparameter
# CONFIG = "../configs/hgflow.yaml"
CONFIG = args.config_model
TRAIN = args.config_train
MODEL = 'refiner' if 'refiner' in CONFIG else 'hgflow'

# Training hyperparameter
BATCH_SIZE = 64
LR = 0.0003
N_EPOCHS = 1000

# Miscellaneous
SEED = 123456
RNG = default_rng(SEED)
pl.seed_everything(SEED)
N_RAY = 0
if N_RAY > 0:
    ray.init(num_cpus=N_RAY,include_dashboard=False)

def get_collate_fn(max_facets, add_indicator=False, max_points=None):

    def pad(x, max_points, pad_value=float('nan')):
        if max_points is None:
            return x
        delta = max_points - x.size(0)
        if delta > 0:
            x = torch.cat([x, torch.full((delta, x.size(1)), pad_value)], dim=0)
        elif delta < 0:
            raise ValueError("max_points is smaller than the number of points!")
        return x

    if not add_indicator:
        def collate_fn(batch):
            points = []
            incidence = []
            for p, i in batch:
                points.append(pad(p, max_points))
                inc = torch.cat([i, torch.zeros(max_facets - i.size(0), i.size(1))],dim=0)
                inc = pad(inc, max_facets, pad_value=0)
                incidence.append(inc)
            return torch.stack(points), torch.stack(incidence)
        return collate_fn
    else:
        def collate_fn(batch):
            points = []
            incidence = []
            for p, i in batch:
                points.append(pad(p, max_points))
                nf = i.size(0)
                inc = torch.cat([i, torch.zeros(max_facets - nf, i.size(1))],dim=0)
                inc = torch.cat([inc, torch.zeros(max_facets, 1)], dim=1)
                inc[:nf,-1] = 1.
                inc = pad(inc, max_facets, pad_value=0)
                incidence.append(inc)
            return torch.stack(points), torch.stack(incidence)
        return collate_fn

train_dataset = ConvexHullData(n_range=N_POINTS,dim=D_FEATS,unit_norm=UNIT_NORM,length=20000)
val_dataset = ConvexHullData(n_range=N_POINTS,dim=D_FEATS,unit_norm=UNIT_NORM,length=2000)

collate_fn = get_collate_fn(max_facets=max(train_dataset.max_facets, 42), add_indicator=ADD_INDICATOR, max_points=N_POINTS[0])
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=0)
valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=0)


if MODEL == "hgflow":
    from lights.hgflow_lightning import HGFlowLightning

    model = HGFlowLightning(
        model_config=CONFIG,
        train_config=TRAIN,
    )
elif MODEL == "refiner":
    from lights.refiner_lightning import IRModel

    model = IRModel(
        model_config=CONFIG,
        train_config=TRAIN,
    )
else:
    raise ValueError("Unknown model type: {}".format(args.model))

data_type = "spherical" if UNIT_NORM else "normal"

logger = WandbLogger(
    name=f"{model.name} P{N_POINTS[0]}to{N_POINTS[-1]}",
    project=f"log_convex_hull_{data_type}",
    log_model=True,
)
checkpoint_callback = ModelCheckpoint(
    monitor='loss/val',
    mode='min',
)

trainer = pl.Trainer(
    accelerator="cuda",
    devices=[0],
    max_epochs=1000,
    check_val_every_n_epoch=1,
    logger=logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, trainloader, valloader)