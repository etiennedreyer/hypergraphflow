import numpy as np
import yaml
import sys
import torch
import torch.nn.functional as F
sys.path.append("../recurrently_predicting_hypergraphs/")
sys.path.append("../HGPflow/")
from functools import partial
from torch.utils.data import Dataset, DataLoader, Subset
from collections import deque, defaultdict

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

        ### Meshing
        elif 'mesh' in self.name:

            self.dataset = MeshDataset(config['input_dir'], 
                                       start=config.get('start', 0), 
                                       stop=config.get('stop', 999_999),
                                       max_faces=config.get('max_faces', None),
                                       max_polygons=config.get('max_polygons', 100),
                                       max_points=config.get('max_points', 1024),
                                       avg_sampled_points_per_face=config.get('avg_sampled_points_per_face', 10)
                                    )
            self.dataset.n_points = [len(v) for v in self.dataset.sampled_points['point_coords']] #self.dataset.vertices]
            self.dataset.max_edges = max(self.dataset.num_polygons)
            self.dataset.max_hhedges = max(self.dataset.num_vertices)

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
            self.dataset.max_edges = self.dataset.max_particles


        ### Delaunay Triangulation
        elif 'delaunay_triangulation' in self.name:
            from delaunay_data import DelaunayTriangulationData

            self.dataset = DelaunayTriangulationData(
                n_range=torch.arange(config['N'][0], config['N'][1]),
                dim=config['D'],
                length=total_size
            )
        else:
            raise NotImplementedError(f"Dataset {self.name} unimplemented.")

        self.max_nodes = max(self.dataset.n_points)
        self.max_edges = self.dataset.max_edges
        self.max_hhedges = getattr(self.dataset, 'max_hhedges', None)

        self.in_feats = config['D']
        if 'particle_flow' not in self.name:
            self.name += f"_{config['D']}D"
        if 'convex_hull' in self.name:
            self.name += f"_{config['N'][0]}to{config['N'][1]-1}"

        ### overwrite max cardinality
        if 'num_nodes' in config:
            if config['num_nodes'] >= self.max_nodes:
                self.max_nodes = config['num_nodes']
            else:
                raise ValueError(
                    f"num_nodes in config is smaller than max in the dataset: "
                    f"{config['num_nodes']} < {self.max_nodes}!"
                )
        if 'num_edges' in config:
            if config['num_edges'] >= self.max_edges:
                self.max_edges = config['num_edges']
            else:
                raise ValueError(
                    f"num_edges in config is smaller than max in the dataset: "
                    f"{config['num_edges']} < {self.max_edges}!"
                )
        if 'num_hhedges' in config:
            if config['num_hhedges'] >= self.dataset.max_hhedges:
                self.max_hhedges = config['num_hhedges']
            else:
                raise ValueError(
                    f"num_hhedges in config is smaller than max in the dataset: "
                    f"{config['num_hhedges']} < {self.dataset.max_hhedges}!"
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

        if 'particle_flow' in self.name:
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

        else:

            def collate_fn(batch, max_edges, pad_nodes=False, max_hhedges=None):

                ### Backward compatibility
                if isinstance(batch[0], tuple):
                    assert len(batch[0]) == 2
                    new_batch = []
                    for tup in batch:
                        new_batch.append({
                            'node_feats': tup[0],
                            'incidence_matrix': tup[1]
                        })
                    batch = new_batch

                max_nodes = max(d['node_feats'].size(0) for d in batch)
                max_edges_batch = max(d['incidence_matrix'].size(0) for d in batch)

                out_dict = {k: [] for k in batch[0].keys()}

                for d in batch:

                    n_edges, n_nodes = d['incidence_matrix'].size()
                    n_pad_edges = max_edges - n_edges
                    n_pad_nodes = max_nodes - n_nodes

                    if pad_nodes:
                        ### Pad to max_nodes
                        nf = F.pad(d['node_feats'], (0, 0, 0, n_pad_nodes), value=torch.nan)
                        im = F.pad(d['incidence_matrix'], (0, n_pad_nodes), value=0)
                        if d.get('node_targets') is not None:
                            nt = F.pad(d['node_targets'], (0, 0, 0, n_pad_nodes), value=torch.nan)
                            out_dict['node_targets'].append(nt)
                        n_nodes = max_nodes

                    ### Pad incidence matrix to max_edges
                    im = torch.cat([im, torch.zeros(n_pad_edges, n_nodes)], dim=0)

                    ### Add indicator "node"
                    im = torch.cat([im, torch.zeros(max_edges, 1)], dim=1)
                    im[:n_edges,-1] = 1.

                    ### Pad edge targets to max_edges
                    if d.get('node_targets') is not None:
                        et = F.pad(d['edge_targets'], (0, 0, 0, n_pad_edges), value=torch.nan)
                        out_dict['edge_targets'].append(et)
                    
                    out_dict['node_feats'].append(nf)
                    out_dict['incidence_matrix'].append(im)

                    if 'vertex_polygon_incidence_matrix' in d:
                        assert max_hhedges is not None, "Please provide max_hhedges for padding vertex_polygon_incidence_matrix."
                        n_hhedges, n_hnodes = d['vertex_polygon_incidence_matrix'].size()
                        n_pad_hhedges = max_hhedges - n_hhedges
                        n_pad_hnodes = max_edges_batch - n_hnodes
                        
                        vp_im = F.pad(d['vertex_polygon_incidence_matrix'], (0, n_pad_hnodes), value=0)
                        if d.get('vertex_targets') is not None:
                            vt = F.pad(d['vertex_targets'], (0, 0, 0, n_pad_hnodes), value=torch.nan)
                            out_dict['vertex_targets'].append(vt)
                        n_hnodes = max_edges_batch
                        vp_im = torch.cat([vp_im, torch.zeros(n_pad_hhedges, n_hnodes)], dim=0)
                        ### add indicator "hnode"
                        vp_im = torch.cat([vp_im, torch.zeros(max_hhedges, 1)], dim=1)
                        vp_im[:n_hhedges,-1] = 1.
                        out_dict['vertex_polygon_incidence_matrix'].append(vp_im)


                    ### Handle other keys
                    for k in d.keys():
                        if k not in ['node_feats', 'incidence_matrix', 'node_targets', 'edge_targets', 
                                     'vertex_polygon_incidence_matrix', 'vertex_targets']:
                            assert isinstance(d[k], torch.Tensor), "Please use tensors for additional batch values."
                            out_dict[k].append(d[k])

                for k in out_dict.keys():
                    out_dict[k] = torch.stack(out_dict[k], dim=0) if len(out_dict[k]) > 0 else None

                return out_dict

            self.collate_fn = partial(collate_fn, max_edges=self.max_edges, pad_nodes=self.pad, 
                                                  max_hhedges=self.max_hhedges)

        # if self.pad:
        #     base_collate_fn = self.collate_fn
        #     def padded_collate_fn(batch):
        #         ID = 'node_feats' if isinstance(batch[0], dict) else 0
        #         pad_until = max(tup[ID].size(0) for tup in batch)
        #         padded_batch = []
        #         p_target = None
        #         for tup in batch:
        #             if isinstance(tup, dict):
        #                 p = tup['node_feats']
        #                 i = tup['incidence_matrix']
        #                 p_target = tup.get('node_targets', None)
        #             elif len(tup) == 2:
        #                 p, i = tup
        #             elif len(tup) == 3:
        #                 p, i, p_target = tup
        #             else:
        #                 raise ValueError("Unexpected batch tuple length.")
        #             p = F.pad(p, (0, 0, 0, pad_until - p.size(0)), value=torch.nan)
        #             i = F.pad(i, (0, pad_until - i.size(1)), value=0)
        #             if p_target is not None:
        #                 p_target = F.pad(p_target, (0, 0, 0, pad_until - p_target.size(0)), value=torch.nan)
        #                 padded_batch.append((p, i, p_target))
        #             else:
        #                 padded_batch.append((p, i))
        #         return base_collate_fn(padded_batch)

        #     self.collate_fn = padded_collate_fn
        
    def get_dataloader(self, dl_config, indices=None):

        ### 1) Get subset of dataset if indices are provided
        dataset = self.dataset
        n_points = self.dataset.n_points
        if indices is not None:
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


import trimesh
import glob
from tqdm import tqdm
import io
from collections import OrderedDict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class MeshDataset(Dataset):
    def __init__(self, input_pattern, 
                 start=0, stop=9999, shuffle=True, 
                 max_faces=9999, max_polygons=100, max_points=2056, 
                 avg_sampled_points_per_face=64, keep_fraction=0.50,
                 random_rotate=True,
                 save_path=None, load_path=None):

        self.max_points = max_points
        self.random_rotate = random_rotate
        self.keep_fraction = keep_fraction

        ### Collect obj files
        inputs = glob.glob(input_pattern)
        if input_pattern.endswith('.txt'):
            print("Extracting merged .obj files from txt file...")
            self.obj_files = self.parse_txt_files(inputs)
        elif input_pattern.endswith('.obj') or input_pattern.endswith('.off'):
            self.obj_files = OrderedDict((f, f) for f in inputs)

        ### Select subset and shuffle
        self.obj_files = {f: obj for i, (f, obj) in enumerate(self.obj_files.items()) if start <= i < stop}
        if shuffle:
            self.obj_files = dict(sorted(self.obj_files.items(), key=lambda item: np.random.rand()))

        ### Load meshes and clean
        self.meshes = [self.get_mesh(obj) for obj in tqdm(self.obj_files.values(), desc="Loading meshes")]
        self.meshes = [self.clean_mesh(mesh) for mesh in tqdm(self.meshes, desc="Cleaning meshes")]
        self.face_to_polygon_maps = []
        self.num_polygons = []
        for mesh in tqdm(self.meshes, desc="Merging coplanar faces"):
            face_map, n_poly = self.merge_coplanar_face_indices(mesh, max_angle_deg=5.0)
            self.face_to_polygon_maps.append(torch.tensor(face_map, dtype=torch.int64))
            self.num_polygons.append(n_poly)

        ### Filter by max faces/polygons
        if max_faces is not None:

            keep_mesh = lambda mesh, Npoly: (
                len(mesh.faces) > 0 and 
                Npoly > 0 and 
                len(mesh.faces) <= max_faces and 
                Npoly <= max_polygons and 
                not (mesh.is_empty or mesh.area < 1e-6)
            )

            indices = [i for i, (mesh, n_poly) in enumerate(zip(self.meshes, self.num_polygons)) if keep_mesh(mesh, n_poly)]
            print(f"Filtering meshes to faces<={max_faces} and polygons<={max_polygons}; keeping {len(indices)}/{len(self.meshes)} meshes.")

            ### apply filtered indices
            self.meshes = [self.meshes[i] for i in indices]
            self.obj_files = {k:v for i, (k,v) in enumerate(self.obj_files.items()) if i in indices}
            self.face_to_polygon_maps = [self.face_to_polygon_maps[i] for i in indices]
            self.num_polygons = [self.num_polygons[i] for i in indices]

        ### Extract vertices
        self.vertices = [torch.tensor(mesh.vertices, dtype=torch.float32) for mesh in tqdm(self.meshes, desc="Extracting vertices")]
        self.vertices = [self.normalize_points(v) for v in self.vertices]

        ### Get perimeter vertices and incidence with polygons
        self.perimeter_vertex_indices = []
        self.perimeter_vertex_coordinates = [] 
        self.vertex_polygon_incidence = []
        self.num_vertices = []
        for mesh, face_to_poly_map in tqdm(zip(self.meshes, self.face_to_polygon_maps), 
                                                total=len(self.meshes), desc="Computing perimeter vertices"):

            perimeter_vertices, perimeter_coords, vertex_poly_incidence = \
                 self.get_perimeter_vertices(mesh, face_to_poly_map)
            self.perimeter_vertex_indices.append(perimeter_vertices)
            self.perimeter_vertex_coordinates.append(perimeter_coords)
            self.vertex_polygon_incidence.append(vertex_poly_incidence)
            self.num_vertices.append(len(perimeter_vertices))

        ### Sample points on surface
        if load_path is None:
            if save_path is None:
                save_path = f"sampled_points_avg{avg_sampled_points_per_face}_polygons{max_polygons}.parquet"
            self.sampled_points = self.sample_points(avg_sampled_points_per_face, save_path=save_path)
        else:
            self.sampled_points = self.load_sampled_points(load_path)


    def sample_points(self, avg_sampled_points_per_face, save_path=None):

        sampled_dict = {
            'point_coords': [], # Float tensor (N_points, 3)
            'point_polygon_indices': [], # Long tensor (N_points,)
            'polygon_normals': [], # Float tensor (N_polygons, 3)
            'polygon_centroids': [], # Float tensor (N_polygons, 3)
        }

        if save_path is not None:
            ### Create parquet writer
            parquet_writer = None
            schema = pa.schema([
                ('point_coords', pa.list_(pa.list_(pa.float32(), 3))),
                ('point_polygon_indices', pa.list_(pa.int64())),
                ('polygon_normals', pa.list_(pa.list_(pa.float32(), 3))),
                ('polygon_centroids', pa.list_(pa.list_(pa.float32(), 3))),
            ])

        for i, mesh in tqdm(enumerate(self.meshes), desc="Sampling surface points", total=len(self.meshes)):

            ### High-density sampling
            points_to_sample = avg_sampled_points_per_face * len(mesh.faces)
            point_coords, face_indices = trimesh.sample.sample_surface(mesh, points_to_sample)
            point_polygon_normals = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)

            ### Normalize points
            point_coords = self.normalize_points(torch.tensor(point_coords, dtype=torch.float32))

            ### Convert face indices to polygon indices
            point_poly_indices = self.face_to_polygon_maps[i][face_indices]

            ### Compute polygon normals and centroids by averaging point-wise values
            # 1) expand indices for scatter
            ppi_expanded = point_poly_indices.unsqueeze(-1).expand(-1, 3)
            N_polys = point_poly_indices.max().item() + 1

            # 2) initialize target tensors
            poly_normals = torch.full(
                (N_polys, 3), 
                float('nan'),
                dtype=point_coords.dtype, 
                device=point_coords.device
            )
            poly_centroids = poly_normals.clone()

            # 3) In-place Scatter Reduce
            poly_normals.scatter_reduce_(
                0, 
                ppi_expanded, 
                point_polygon_normals, 
                reduce='mean', 
                include_self=False
            )

            poly_centroids.scatter_reduce_(
                0, 
                ppi_expanded, 
                point_coords, 
                reduce='mean', 
                include_self=False
            )

            ### Store
            sampled_dict['point_coords'].append(point_coords)
            sampled_dict['point_polygon_indices'].append(point_poly_indices)
            sampled_dict['polygon_normals'].append(poly_normals)
            sampled_dict['polygon_centroids'].append(poly_centroids)

            if save_path is not None:
                ### Create parquet writer
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(save_path, schema)

                ### Create arrow table
                row_data = {
                        'point_coords': pa.array(
                            [sampled_dict['point_coords'][-1].numpy().tolist()],
                            type=pa.list_(pa.list_(pa.float32(), 3))
                        ),
                        'point_polygon_indices': pa.array(
                            [sampled_dict['point_polygon_indices'][-1].numpy().tolist()], 
                            type=pa.list_(pa.int64())
                        ),
                        'polygon_normals': pa.array(
                            [sampled_dict['polygon_normals'][-1].numpy().tolist()], 
                            type=pa.list_(pa.list_(pa.float32(), 3))
                        ),
                        'polygon_centroids': pa.array(
                            [sampled_dict['polygon_centroids'][-1].numpy().tolist()], 
                            type=pa.list_(pa.list_(pa.float32(), 3))
                        ),
                    }

                table = pa.Table.from_pydict(row_data, schema=schema)
                parquet_writer.write_table(table)

        if save_path is not None:
            parquet_writer.close()
            print(f"Saved sampled points to {save_path}.")

        return sampled_dict

    def load_sampled_points(self, load_path): # TODO: check

        sampled_dict = {
            'point_coords': [],
            'point_polygon_indices': [],
            'polygon_normals': [],
            'polygon_centroids': [],
        }

        ### Read parquet file
        table = pq.read_table(load_path)
        df = table.to_pandas()

        for i in tqdm(range(len(df)), desc=f"Loading sampled points from parquet {load_path}"):

            sampled_dict['point_coords'].append(torch.tensor(df['point_coords'][i], dtype=torch.float32))
            sampled_dict['point_polygon_indices'].append(torch.tensor(df['point_polygon_indices'][i], dtype=torch.int64))
            sampled_dict['polygon_normals'].append(torch.tensor(df['polygon_normals'][i], dtype=torch.float32))
            sampled_dict['polygon_centroids'].append(torch.tensor(df['polygon_centroids'][i], dtype=torch.float32))

        return sampled_dict


    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):

        point_coords = self.sampled_points['point_coords'][idx]
        point_poly_indices = self.sampled_points['point_polygon_indices'][idx]
        poly_normals = self.sampled_points['polygon_normals'][idx]
        poly_centroids = self.sampled_points['polygon_centroids'][idx]

        ### point-wise polygon targets
        point_poly_normals = poly_normals[point_poly_indices]
        point_poly_centroids = poly_centroids[point_poly_indices]

        ### perimeter vertices, their coordinates, and incidence matrix with polygons
        perimeter_vertex_coords = self.perimeter_vertex_coordinates[idx]
        vertex_poly_incidence = self.vertex_polygon_incidence[idx]

        if 0.0 < self.keep_fraction < 1.0:

            ### Randomly select a subset of up to max_points sampled points
            N_points = min(int(self.keep_fraction * point_coords.shape[0]), self.max_points)
            subset = torch.randperm(point_coords.shape[0])[:N_points]
            subset = subset.sort().values
            point_coords = point_coords[subset]
            point_poly_indices = point_poly_indices[subset]
            point_poly_normals = point_poly_normals[subset]
            point_poly_centroids = point_poly_centroids[subset]

            ### Get contiguous polygon indices
            point_poly_indices, is_valid_poly, is_valid_point = self.get_contiguous_polygon_indices(point_poly_indices)

            if not is_valid_point.all():
                ### Drop points that belong to polygons with insufficient points
                point_coords = point_coords[is_valid_point]
                point_poly_normals = point_poly_normals[is_valid_point]
                point_poly_centroids = point_poly_centroids[is_valid_point]

            if not is_valid_poly.all():
                ### Drop invalid polygons from perimeter vertices and incidence matrix
                vertex_poly_incidence = vertex_poly_incidence[:, is_valid_poly]
                is_valid_vertex = vertex_poly_incidence.any(dim=1)
                perimeter_vertex_coords = perimeter_vertex_coords[is_valid_vertex]
                vertex_poly_incidence = vertex_poly_incidence[is_valid_vertex, :]

        ### Compute incidence matrices between points and polygons          
        incidence_matrix = self.get_incidence_matrix(point_poly_indices)

        ### Random rotation augmentation
        if self.random_rotate:
            matrix = trimesh.transformations.random_rotation_matrix()[:3,:3]
            point_coords = point_coords @ torch.tensor(matrix, dtype=torch.float32).T
            point_poly_normals = point_poly_normals @ torch.tensor(matrix, dtype=torch.float32).T
            point_poly_centroids = point_poly_centroids @ torch.tensor(matrix, dtype=torch.float32).T
            perimeter_vertex_coords = perimeter_vertex_coords @ torch.tensor(matrix, dtype=torch.float32).T

        ### Copy over the polygon targets
        point_poly_targets = torch.cat([point_poly_normals, point_poly_centroids], dim=-1)
        N_polygons = point_poly_indices.max().item() + 1
        polygon_targets = torch.full((N_polygons, 3+3), float('nan'), dtype=torch.float32)
        polygon_targets[point_poly_indices] = point_poly_targets
        assert not torch.isnan(polygon_targets).any(), "Some polygons have no targets assigned!"

        out_dict = {
            'mesh_idx': torch.tensor(idx, dtype=torch.int64),
            'num_nodes': torch.tensor(point_coords.shape[0], dtype=torch.int64),
            'num_polygons': torch.tensor(N_polygons, dtype=torch.int64),
            'num_vertices': torch.tensor(perimeter_vertex_coords.shape[0], dtype=torch.int64),
            'node_feats': point_coords,
            'incidence_matrix': incidence_matrix,
            'node_targets': point_poly_normals,
            'edge_targets': polygon_targets,
            'vertex_polygon_incidence_matrix': vertex_poly_incidence,
            'vertex_targets': perimeter_vertex_coords,
        }
        return out_dict

    def parse_txt_files(self, txt_files):
        obj_files = {}
        for txt_file in txt_files:
            file_id = txt_file.split('_')[-1].split('.txt')[0]
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('newfile'):
                    obj_file = line.split(' ')[1].strip()
                    obj_content = []
                else:
                    obj_content.append(line)
                if i == len(lines) - 1 or lines[i+1].startswith('newfile'):
                    obj_data = ''.join(obj_content)
                    obj_files[f"{obj_file}"] = io.StringIO(obj_data)

        print(f"Parsed {len(obj_files)} .obj files from {len(txt_files)} .txt files.")
        return obj_files

    def normalize_points(self, vertices):
        centroid = vertices.mean(dim=0, keepdim=True)
        vertices = vertices - centroid
        scale = torch.norm(vertices, dim=-1).max()
        if scale < 1e-6:
            scale = 1.0
        vertices = vertices / scale
        return vertices

    @staticmethod
    def get_mesh(obj_file, surface_idx=9999):
    
        scene = trimesh.load(obj_file, file_type='obj', force='mesh')

        if isinstance(scene, trimesh.Scene):
            meshes = scene.dump()
            if surface_idx < len(meshes):
                mesh = meshes[surface_idx]
            else:
                mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene

        return mesh

    @staticmethod
    def clean_mesh(mesh):
        mesh.remove_infinite_values()
        mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=3)
        mesh.remove_duplicate_faces()
        mesh.process(validate=True)
        mesh.fix_normals()
        return mesh

    @staticmethod
    def merge_coplanar_face_indices(mesh, max_angle_deg=5.0):

        adj_face_pairs = mesh.face_adjacency # returns list of face index pairs that are adjacent, without reverses
        adj_angles = mesh.face_adjacency_angles # same shape but gives angles for each pair
        coplanar_mask = adj_angles < np.deg2rad(max_angle_deg)
        adj_cop_face_pairs = adj_face_pairs[coplanar_mask]

        n_faces = len(mesh.faces)
        adj_lists = [[] for _ in range(n_faces)]
        for f1, f2 in adj_cop_face_pairs:
            adj_lists[f1].append(f2)
            adj_lists[f2].append(f1)

        face_to_polygon_map = [-1 for _ in range(n_faces)]
        current_polygon_idx = 0

        for face_idx in range(n_faces):

            ### Already assigned
            if face_to_polygon_map[face_idx] != -1:
                continue

            ### Loners
            elif len(adj_lists[face_idx]) == 0:
                face_to_polygon_map[face_idx] = current_polygon_idx

            ### New polygon (breadth-first search)
            else:
                face_to_polygon_map[face_idx] = current_polygon_idx
                q = deque([face_idx])
                while q:
                    current_face = q.popleft()
                    ### Loop over neighbors
                    for neighbor in adj_lists[current_face]:
                        ### Check if the neighbor has already been assigned
                        if face_to_polygon_map[neighbor] == -1:
                            face_to_polygon_map[neighbor] = current_polygon_idx
                            q.append(neighbor)

            current_polygon_idx += 1

        num_polygons = current_polygon_idx

        return np.array(face_to_polygon_map, dtype=np.int64), num_polygons

    @staticmethod
    def get_contiguous_polygon_indices(poly_indices, min_points_per_polygon=3):

        '''
        Inputs:
            poly_indices: tensor of shape [num_points], polygon index for each point
            min_points_per_polygon: int, minimum number of points required for a polygon to be valid
        Outputs:
            new_poly_indices: tensor of shape [num_valid_points], new contiguous polygon indices for valid points
            is_valid_point: tensor of shape [num_points], boolean mask indicating valid points
        '''

        ### For finite sampled points, small polygons may have low sampled points --> require >=3
        _unique, inverse_indices, counts = torch.unique(poly_indices, return_inverse=True, return_counts=True)
        is_valid_poly = counts >= min_points_per_polygon

        ### Mask for sampled points
        is_valid_point = is_valid_poly[inverse_indices]

        ### New contiguous polygon indices (i.e. [F, T, T, F, T] --> [0, 1, 2, 2, 3])
        poly_index_map = torch.cumsum(is_valid_poly, dim=0) - 1
        new_poly_indices = poly_index_map[inverse_indices[is_valid_point]]

        assert new_poly_indices.max().item() + 1 == is_valid_poly.sum().item()

        return new_poly_indices, is_valid_poly, is_valid_point

    @staticmethod
    def get_perimeter_vertices(mesh, face_to_polygon_map, min_corner_angle_deg=15.0):

        num_polygons = int(face_to_polygon_map.max()) + 1
        num_tri_vertices = len(mesh.vertices)
        vertex_poly_incidence = torch.zeros((num_tri_vertices, num_polygons), dtype=torch.bool)

        for poly_idx in range(num_polygons):

            ### A list of 3-vertex indices
            face_mask = torch.where(face_to_polygon_map == poly_idx)[0]
            poly_faces = mesh.faces[face_mask.numpy()]

            if len(poly_faces) == 0:
                raise ValueError(f"Polygon {poly_idx} has no faces!")

            ### A list of 2-vertex edges
            edges = np.concatenate([
                poly_faces[:, [0, 1]],
                poly_faces[:, [1, 2]],
                poly_faces[:, [2, 0]],
            ], axis=0)

            ### reversed edges are identical
            edges = np.sort(edges, axis=1)

            unique_edge, counts = np.unique(edges, axis=0, return_counts=True)

            ### Edges internal to the polygon appear twice, perimeter edges only once
            perimeter_edges = unique_edge[counts == 1]

            ### Shake out and count how often each vertex appears in perimeter edges
            p_verts_flat = perimeter_edges.flatten()
            v_unique, v_counts = np.unique(p_verts_flat, return_counts=True)

            ### Identify vertices that connect exactly two perimeter edges
            is_standard = v_counts == 2
            v_standard = v_unique[is_standard]

            ### Opt to keep non-standard ones by default
            v_junction = v_unique[~is_standard]

            ### Next, for the standard vertices, keep only those that open by more than 5 degrees
            if len(v_standard) > 0:
                edge_is_standard_mask = np.any(np.isin(perimeter_edges, v_standard), axis=1)

                e_standard = perimeter_edges[edge_is_standard_mask]

                ### Construct a list of edges with the goal of placing neighboring edges side-by-side
                e_pairs = np.concatenate([
                    e_standard,              # [[a, b], [b, c], ... [a, e]]
                    e_standard[:, [1, 0]]    # [[b, a], [c, b], ... [e, a]]  
                ], axis=0)

                ### We only need the angles corresponding to standard vertices on those edges
                e_pairs = e_pairs[ np.isin(e_pairs[:, 0], v_standard) ]

                ### Arange so that edges sharing a common src/dst vertex are adjacent
                e_pairs = e_pairs[ np.argsort(e_pairs[:, 0]) ]
                ### At this point there should be exactly two entries per src vertex

                ### Extract [c, a], [c, b] ==> c, a, b vertices for angle computation
                ### Note: corners are a->c->b with (c for center)
                c_verts = e_pairs[0::2, 0]
                a_verts = e_pairs[0::2, 1]
                b_verts = e_pairs[1::2, 1]

                c_coords = mesh.vertices[c_verts]
                a_coords = mesh.vertices[a_verts]
                b_coords = mesh.vertices[b_verts]

                vec_ac = c_coords - a_coords
                vec_cb = b_coords - c_coords

                len_ac = np.sqrt((vec_ac**2).sum(axis=1, keepdims=True)) + 1e-8
                len_cb = np.sqrt((vec_cb**2).sum(axis=1, keepdims=True)) + 1e-8

                ### Dot prod = element-wise multiplication, then sum over spatial dims
                dot_ac_cb = ((vec_ac / len_ac) * (vec_cb / len_cb)).sum(axis=1)

                ### Require angle > 5 degrees ==> cos(angle) < cos(5 deg)
                valid_corners = dot_ac_cb < np.cos(np.deg2rad(min_corner_angle_deg))
                v_standard = c_verts[valid_corners]

            perimeter_vertices_local = np.concatenate([v_standard, v_junction], axis=0)

            ### Update incidence matrix
            vertex_poly_incidence[perimeter_vertices_local, poly_idx] = True

        ### Keep only perimeter vertices
        perimeter_vertices_global = torch.where(vertex_poly_incidence.any(dim=1))[0]
        vertex_poly_incidence = vertex_poly_incidence[perimeter_vertices_global, :]

        ### Extract coordinates of these perimeter vertices
        perimeter_coords = torch.tensor(mesh.vertices[perimeter_vertices_global.numpy()], dtype=torch.float32)

        return perimeter_vertices_global, perimeter_coords, vertex_poly_incidence

    @staticmethod
    def get_incidence_matrix(face_indices):
        num_vertices = face_indices.shape[0]
        num_faces = face_indices.max().item() + 1

        incidence_matrix = torch.zeros((num_faces, num_vertices), dtype=torch.bool)

        row_indices = face_indices
        col_indices = torch.arange(num_vertices)
        incidence_matrix[row_indices, col_indices] = True

        return incidence_matrix