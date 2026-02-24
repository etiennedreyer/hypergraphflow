import numpy as np
import yaml
import sys
import torch
import torch.nn.functional as F
sys.path.append("../recurrently_predicting_hypergraphs/")
sys.path.append("../HGPflow/")
from torch.utils.data import DataLoader, Subset

class HyperGraphDataset:

    def __init__(self, config, total_size):

        if type(config) is str:
            with open(config, 'r') as f:
                config = yaml.safe_load(f)

        self.config = config
        self.name = config['name']
        self.add_indicator = config.get('add_indicator', True)
        self.collate_fn = None
        self.sampler = None
        self.pad = True
        self.total_size = total_size

        ### Convex Hull
        if 'convex_hull' in self.name:
            from convex_hull_dataset import ConvexHullData

            self.dataset = ConvexHullData(
                n_range=torch.arange(config['N'][0], config['N'][1]),
                dim=config['D'],
                unit_norm=config.get('norm'),
                length=total_size
            )
            self.name += "_spherical" if config['norm'] else "_normal"

        ### Particle Flow
        elif 'particle_flow' in self.name:

            ds_kwargs = {
                    'filename': config['filename'],
                    'config_v': config,
                    'reduce_ds': -1,
                    'compute_incidence': True}

            from hgpflow_v2.dataset.dataset_mini import PflowDatasetMini
            self.dataset = PflowDatasetMini(**ds_kwargs)
            self.dataset.n_points = self.dataset.n_nodes
            self.dataset.max_facets = self.dataset.max_particles


        ### Delaunay Triangulation
        elif 'delaunay_triangulation' in self.name:
            from delaunay_data import DelaunayTriangulationData

            self.dataset = DelaunayTriangulationData(
                n_range=torch.arange(config['N'][0], config['N'][1]),
                dim=config['D'],
                length=total_size
            )
        ### SET Game
        elif 'set_game' in self.name:
            from utils.set_game_data import SetGameData
            self.dataset = SetGameData()
            self.dataset.n_points = [config['N']]
            self.dataset.max_facets = config['K']
        else:
            raise NotImplementedError(f"Dataset {self.name} unimplemented.")

        self.max_nodes = max(self.dataset.n_points)
        self.max_edges = self.dataset.max_facets

        self.in_feats = config['D']
        if 'convex_hull' in self.name:
            self.name += f"_{config['D']}D"
            self.name += f"_{config['N'][0]}to{config['N'][1]-1}"

        ### overwrite max cardinality
        if 'num_edges' in config:
            if config['num_edges'] >= self.max_edges:
                self.max_edges = config['num_edges']
            else:
                raise ValueError(
                    f"num_edges in config is smaller than max in the dataset: "
                    f"{config['num_edges']} < {self.max_edges}!"
                )
        if 'num_nodes' in config:
            if config['num_nodes'] >= self.max_nodes:
                self.max_nodes = config['num_nodes']
            else:
                raise ValueError(
                    f"num_nodes in config is smaller than max in the dataset: "
                    f"{config['num_nodes']} < {self.max_nodes}!"
                )


    def get_sampler(self, dataset=None, n_points=None):

        if dataset is None:
            dataset = self.dataset
        if n_points is None:
            n_points = self.dataset.n_points

        if 'convex_hull' in self.name:
            from convex_hull_dataset import BucketSampler

            self.sampler = BucketSampler(dataset, self.batch_size, 
                                    n_points,
                                    shuffle=self.shuffle)
            
        elif 'particle_flow' in self.name:

            from hgpflow_v2.dataset.dataset_mini import PflowSamplerMini
            self.sampler = PflowSamplerMini(np.array(n_points),
                                    batch_size=self.batch_size, 
                                    remove_idxs=True) # TODO: check remove_idxs

    def get_collate_fn(self):

        if ('convex_hull' in self.name) or ('delaunay_triangulation' in self.name):
            from convex_hull_dataset import get_collate_fn
            self.collate_fn = get_collate_fn(self.max_edges, 
                                                add_indicator=self.add_indicator)

        elif 'particle_flow' in self.name:
            from hgpflow_v2.dataset.dataset_mini import collate_fn_mini

            def custom_collate_fn(batch):
                batch = collate_fn_mini(batch)

                ### prepare incidence matrix
                inc = batch['incidence_truth']
                ind = batch['indicator_truth']
                im = torch.cat([inc, ind.unsqueeze(-1)], dim=-1)

                ### prepare node features
                node_feats = batch['node']['skip_feat0']
                is_track = batch['node']['is_track']
                node_feats = torch.cat([node_feats, is_track.unsqueeze(-1)], dim=-1)

                return node_feats, im

            self.collate_fn = custom_collate_fn
        
        elif 'set_game' in self.name:
            self.pad = False
            from utils.set_game_data import get_collate_fn
            self.collate_fn = get_collate_fn(self.max_edges)

        if self.pad:
            base_collate_fn = self.collate_fn
            def padded_collate_fn(batch):
                pad_until = max(p.size(0) for p, _ in batch)
                padded_batch = []
                for p, i in batch:
                    p = F.pad(p, (0, 0, 0, pad_until - p.size(0)), value=torch.nan)
                    i = F.pad(i, (0, pad_until - i.size(1)), value=0)
                    padded_batch.append((p, i))
                return base_collate_fn(padded_batch)

            self.collate_fn = padded_collate_fn
        
    def get_dataloader(self, dl_config, indices=None):

        ### 1) Get subset of dataset if indices are provided
        dataset = self.dataset
        n_points = self.dataset.n_points
        if indices is not None and len(indices) > 0:
            dataset = Subset(self.dataset, indices)
            n_points = [self.dataset.n_points[i] for i in indices]

        self.shuffle = dl_config['shuffle']
        self.batch_size = dl_config['batch_size']
        ### 2) Get sampler
        if dl_config.get('sampler', False):
            self.pad = False
            if self.batch_size > 1:
                self.get_sampler(dataset, n_points=n_points)

        ### 3) Get collate function
        self.get_collate_fn()

        return DataLoader(
            dataset,
            shuffle=self.shuffle if not self.sampler else False,
            batch_size=self.batch_size if not self.sampler else 1,
            batch_sampler=self.sampler,
            collate_fn=self.collate_fn,
            num_workers=dl_config['num_workers'],
            pin_memory=True,
            persistent_workers=True if dl_config['num_workers'] > 0 else False,
        )