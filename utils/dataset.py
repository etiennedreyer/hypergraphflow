import numpy as np
import yaml
import sys
import torch
import torch.nn.functional as F
sys.path.append("../recurrently_predicting_hypergraphs/")
sys.path.append("../HGPflow/")
from torch.utils.data import Dataset, DataLoader, Subset
from collections import deque

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
            self.dataset.n_points = [len(v) for v in self.dataset.sampled_points] #self.dataset.vertices]
            self.dataset.max_facets = max(self.dataset.num_sampled_polygons)

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
        else:
            raise NotImplementedError(f"Dataset {self.name} unimplemented.")

        self.max_nodes = max(self.dataset.n_points)
        self.max_edges = self.dataset.max_facets

        self.in_feats = config['D']
        if 'particle_flow' not in self.name:
            self.name += f"_{config['D']}D"
        if 'convex_hull' in self.name:
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
            from convex_hull_dataset import get_collate_fn
            self.collate_fn = get_collate_fn(self.max_edges, 
                                                add_indicator=self.add_indicator)

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

class MeshDataset(Dataset):
    def __init__(self, input_pattern, start=0, stop=9999, max_faces=9999, max_polygons=100, max_points=1024, shuffle=True, avg_sampled_points_per_face=10):

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
            self.face_to_polygon_maps.append(face_map)
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

        ### Sample surface points
        self.sampled_points = []
        self.sampled_polygons = []
        self.num_sampled_polygons = []

        for i, mesh in tqdm(enumerate(self.meshes), desc="Sampling surface points", total=len(self.meshes)):

            ### Sampling
            points_to_sample = avg_sampled_points_per_face * len(mesh.faces)
            if max_points is not None:
                points_to_sample = min(points_to_sample, max_points)
            point_coords, face_indices = trimesh.sample.sample_surface(mesh, points_to_sample)

            ### Convert face indices to polygon indices
            poly_indices = self.face_to_polygon_maps[i][face_indices]

            ### For finite sampled points, small polygons may have low sampled points --> require >=3
            _unique, inverse_indices, counts = np.unique(poly_indices, return_inverse=True, return_counts=True)
            is_valid_poly = counts >= 3

            ### Mask for sampled points
            is_valid_point = is_valid_poly[inverse_indices]

            ### New contiguous polygon indices (i.e. [F, T, T, F, T] --> [0, 1, 2, 2, 3])
            poly_index_map = np.cumsum(is_valid_poly) - 1

            ### Apply masks
            point_coords = point_coords[is_valid_point]
            poly_indices = poly_index_map[inverse_indices[is_valid_point]]

            ### Store
            self.sampled_points.append(torch.tensor(point_coords, dtype=torch.float32))
            self.sampled_polygons.append(torch.tensor(poly_indices, dtype=torch.int64))
            self.num_sampled_polygons.append(is_valid_poly.sum())

        ### Extract vertices, faces, incidence matrices
        # self.vertices = [torch.tensor(mesh.vertices, dtype=torch.float32) for mesh in tqdm(self.meshes, desc="Extracting vertices")]
        # self.vertices = [self.normalize_vertices(v) for v in self.vertices]
        self.sampled_points = [self.normalize_vertices(v) for v in self.sampled_points]
        # self.faces = [torch.tensor(mesh.faces, dtype=torch.int64) for mesh in tqdm(self.meshes, desc="Extracting faces")]
        # self.incidence_matrices = [self.get_incidence_matrix(f, len(v)) 
        #                             for v, f in tqdm(zip(self.vertices, self.faces),
        #                                              desc="Computing incidence matrices", 
        #                      
        self.incidence_matrices = [self.get_incidence_matrix(idxs) for idxs in tqdm(self.sampled_polygons, desc="Computing incidence matrices")]

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):

        return self.sampled_points[idx], self.incidence_matrices[idx]

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

    def normalize_vertices(self, vertices):
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
        mesh.process(validate=True)
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

    # @staticmethod
    # def get_incidence_matrix(faces, num_vertices):

    #     num_faces = faces.shape[0]
    #     deg_faces = faces.shape[1]

    #     incidence_matrix = torch.zeros((num_faces, num_vertices), dtype=torch.bool)

    #     row_indices = torch.arange(num_faces).unsqueeze(1).repeat(1, deg_faces).flatten()
    #     col_indices = faces.flatten()

    #     incidence_matrix[row_indices, col_indices] = True
    #     return incidence_matrix

    @staticmethod
    def get_incidence_matrix(face_indices):
        num_vertices = face_indices.shape[0]
        num_faces = face_indices.max().item() + 1

        incidence_matrix = torch.zeros((num_faces, num_vertices), dtype=torch.bool)

        row_indices = face_indices
        col_indices = torch.arange(num_vertices)
        incidence_matrix[row_indices, col_indices] = True

        return incidence_matrix