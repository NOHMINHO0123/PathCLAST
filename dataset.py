import os
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torchvision import transforms
from torchtoolbox.transform import Cutout
import cv2
import scipy
import scanpy as sc
import networkx as nx
from collections import defaultdict

from utils import load_ST_file, build_her2st_data


class PathwayProcessor:
    """For building and updating pathway graphs."""
    def __init__(self, csv_file):
        self.prebuilt_graphs = {}  
        self.df = pd.read_csv(csv_file)  
        self.df = self.df[self.df['EdgeInfo'].notnull()]

    def create_static_pathway_graphs(self):
        for pathway_id in self.df['PathwayID'].unique():
            pathway_data = self.df[self.df['PathwayID'] == pathway_id]

            # Create pathwaygraph
            G = nx.Graph()
            node_info = pathway_data['NodeInfo'].iloc[0]
            nodes = [node.strip() for node in node_info.split(';')]
            G.add_nodes_from(nodes)
            edge_info = pathway_data['EdgeInfo'].iloc[0]
            edges = [
                tuple(edge.strip('()').split(', '))
                for edge in edge_info.split(';') if edge.strip()
            ]
            G.add_edges_from(edges)
            self.prebuilt_graphs[pathway_id] = G
            
    def update_graph_with_expression(self, sPathID, expression_data_matrix, symbol_to_index):
        """
        패스웨이 그래프 내 유전자 발현 데이터를 그대로 추가하는 메서드.
        노드 피처로 이미 정규화된 발현 데이터를 사용합니다.
        """

        if sPathID not in self.prebuilt_graphs:
            return None
        G = deepcopy(self.prebuilt_graphs[sPathID])
        pathway_genes = [node for node in G.nodes() if node in symbol_to_index]

        if len(pathway_genes) == 0:
            return None  # or handle the empty case appropriately
        
        pathway_expression_data = expression_data_matrix[:, [symbol_to_index[gene] for gene in pathway_genes]]
        for node in G.nodes():
            gene_symbol = node  
            if gene_symbol in symbol_to_index:
                idx = pathway_genes.index(gene_symbol)
                expression_data = pathway_expression_data[:, idx]
                
                G.nodes[node]['expression_data'] = expression_data
            else:
                G.nodes[node]['expression_data'] = np.zeros(expression_data_matrix.shape[0], dtype=np.float32)

        return G

class Dataset(data.Dataset):
    
    def __init__(self, dataset, path, name, csv_file,
                 prob_node_drop=0.5, pct_node_drop=0.1, 
                 prob_edge_perturb=0.5, pct_edge_perturb=0.1, img_size=112, train=True):
        
        self.dataset = dataset
        self.processor = PathwayProcessor(csv_file)
        self.processor.create_static_pathway_graphs()
        self.train = train
    
        # Load data
        if dataset == "SpatialLIBD":
            adata = load_ST_file(os.path.join(path, name))
            adata.X = adata.X.A
            df_meta = pd.read_csv(os.path.join(path, name, 'metadata.tsv'), sep='\t')
            self.label = pd.Categorical(df_meta['layer_guess']).codes
            full_image = cv2.imread(os.path.join(path, name, f'{name}_full_image.tif'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            patches = []
            for x, y in adata.obsm['spatial']:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                patches.append(full_image[y-img_size:y+img_size, x-img_size:x+img_size])
            patches = np.array(patches)
            self.image = patches

        elif dataset == "Her2st":
            adata, patches = build_her2st_data(path, name, img_size)
            self.label = adata.obs['label']
            self.image = patches

        elif dataset == "IDC":
            adata = load_ST_file(os.path.join(path, name))
            adata.X = adata.X.A
            self.label = np.zeros(adata.shape[0], dtype=int)

            full_image = cv2.imread(os.path.join(path, name, f'{name}.tif'))
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
            patches = []
            for x, y in adata.obsm['spatial']:
                patches.append(full_image[y - img_size:y + img_size, x - img_size:x + img_size])
            patches = np.array(patches)
            self.image = patches

        self.n_clusters = self.label.max() + 1
        self.spatial = adata.obsm['spatial']
        self.n_pos = self.spatial.max() + 1
        
        self.gene = adata
        self.gene = self.gene[self.label != -1]      
        self.image = self.image[self.label != -1]
        self.label = self.label[self.label != -1]
        


        
        expression_idx = list(adata.var_names)
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(expression_idx)}
        
        self.train = train
        self.img_train_transform = transforms.Compose([
            Cutout(0.5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.img_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.graph_train_transform = GraphGeneTransforms(prob_node_drop=0.5, pct_node_drop=0.1, 
                                                         prob_edge_perturb=0.5, pct_edge_perturb=0.1)
            
    def update_graphs_with_expression(self, xg):
        updated_graphs = {}
        
        for sPathID, graph in self.processor.prebuilt_graphs.items():
            if self.dataset == "SpatialLIBD":
                pdPathwayExprMatrix = xg.X.astype(float)
            elif self.dataset == "Her2st":
                pdPathwayExprMatrix = xg.X.astype(int)
            elif self.dataset == "IDC":
                pdPathwayExprMatrix = xg.X.astype(float)


            
            updated_graph = self.processor.update_graph_with_expression(sPathID, pdPathwayExprMatrix, self.symbol_to_index)
            if updated_graph:
                updated_graphs[sPathID] = updated_graph

        return updated_graphs
     
    def convert_graphs_to_tensor(self, graphs):
        """패스웨이 그래프를 torch_geometric Data 객체로 변환"""
        pyg_data_objects = []
        
        for sPathID, G in graphs.items():
            node_indices = {node: i for i, node in enumerate(G.nodes())}
            edge_index = torch.tensor([[node_indices[u], node_indices[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

            node_features = []
            for node in G.nodes():
                expression_data = G.nodes[node].get('expression_data', np.zeros(1)).reshape(-1, 1)
                node_features.append(expression_data)
            node_features = torch.tensor(node_features, dtype=torch.float).squeeze()
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(1)

            pyg_data_objects.append(Data(x=node_features, edge_index=edge_index))
            
        
        return pyg_data_objects
        



    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        spatial = torch.from_numpy(self.spatial[idx])
        y = self.label[idx]
        xg = self.gene[idx]
        
        updated_graphs = self.update_graphs_with_expression(xg)
        xg = self.convert_graphs_to_tensor(updated_graphs)

        if self.train:
            
            xg_u = [self.graph_train_transform(deepcopy(graph)) for graph in xg]
            xg_v = [self.graph_train_transform(deepcopy(graph)) for graph in xg]
            xi_u = self.img_train_transform(self.image[idx])
            xi_v = self.img_train_transform(self.image[idx])

            return xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx
        else: 
            xi = self.img_test_transform(self.image[idx])
            return xg, xi, spatial, y, idx





class GraphGeneTransforms(nn.Module):
    """Graph augmentation: node‑dropping & edge‑perturbation."""
    def __init__(self, prob_node_drop=0.5, pct_node_drop=0.1, 
                 prob_edge_perturb=0.5, pct_edge_perturb=0.1):  
        super(GraphGeneTransforms, self).__init__()
        self.prob_node_drop = prob_node_drop  
        self.pct_node_drop = pct_node_drop  
        self.prob_edge_perturb = prob_edge_perturb 
        self.pct_edge_perturb = pct_edge_perturb  

    def forward(self, data):
        """Return torch_geometric.data.Data"""
        xg = data.x  
        num_nodes = xg.size(0) 
        edge_index = data.edge_index  

        # node dropping
        if torch.rand(1) < self.prob_node_drop:
            drop_num = int(num_nodes * self.pct_node_drop)
            drop_indices = torch.randperm(num_nodes)[:drop_num]
            keep_indices = torch.tensor([i for i in range(num_nodes) if i not in drop_indices])

            xg = xg[keep_indices]
            node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices.tolist())}
            new_edge_index = [
                (node_map[src.item()], node_map[dst.item()]) for src, dst in edge_index.t()
                if src.item() in node_map and dst.item() in node_map
            ]
            if new_edge_index:
                edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            keep_indices = torch.arange(num_nodes)
            
        # edge perturbation
        if edge_index.size(1) > 0 and torch.rand(1) < self.prob_edge_perturb:
            edge_num = edge_index.size(1)
            edge_perturb_num = int(edge_num * self.pct_edge_perturb)
            edge_indices_to_remove = torch.randperm(edge_num)[:edge_perturb_num]
            edge_indices_to_keep = torch.tensor([i for i in range(edge_num) if i not in edge_indices_to_remove])
            edge_index = edge_index[:, edge_indices_to_keep]
            new_edges = torch.randint(0, len(keep_indices), (2, edge_perturb_num))
            edge_index = torch.cat([edge_index, new_edges], dim=1)

        data.x = xg
        data.edge_index = edge_index
        return data


