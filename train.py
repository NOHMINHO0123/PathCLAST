
from ast import parse
import os
import random
import numpy as np
import scanpy as sc
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.data import Batch as GeoBatch
from torch.utils.data import DataLoader
import argparse

from dataset import Dataset
from model import SpaCLR, TrainerSpaCLR
from utils import get_predicted_results
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def custom_collate_fn(batch):
    if len(batch[0]) == 8:
        xg_list, xg_u_list, xg_v_list, xi_u, xi_v, spatial, y, idx_returned = zip(*batch)

        xg = [GeoBatch.from_data_list(sublist) for sublist in xg_list]
        xg_u = [GeoBatch.from_data_list(sublist) for sublist in xg_u_list]
        xg_v = [GeoBatch.from_data_list(sublist) for sublist in xg_v_list]
        xi_u = torch.stack(xi_u, dim=0)
        xi_v = torch.stack(xi_v, dim=0)
        spatial = torch.stack(spatial, dim=0)
        y = torch.tensor(y, dtype=torch.long)
        idx_returned = torch.tensor(idx_returned, dtype=torch.long)
        return  xg, xg_u, xg_v, xi_u, xi_v, spatial, y, idx_returned
    
    else:
        xg_list, xi, spatial, y, idx_returned = zip(*batch)

        xg = [GeoBatch.from_data_list(sublist) for sublist in xg_list]
        xi = torch.stack(xi, dim=0)
        spatial = torch.stack(spatial, dim=0)
        y = torch.tensor(y, dtype=torch.long)
        idx_returned = torch.tensor(idx_returned, dtype=torch.long)
        return xg, xi, spatial, y, idx_returned
    
def train(args, name):
    # seed
    seed_torch(1)
    
    # args
    path = args.path
    last_dim = args.last_dim
    hidden_dim = args.hidden_dim
    image_dims = [last_dim]
    csv_file = args.csv_file
    lr = args.lr
    p_drop = args.p_drop
    batch_size = args.batch_size
    dataset = args.dataset
    epochs = args.epochs
    img_size = args.img_size
    device = args.device
    log_name = args.log_name
    num_workers = args.num_workers
    prob_edge_perturb = args.prob_edge_perturb
    pct_edge_perturb = args.pct_edge_perturb
    prob_node_drop = args.prob_node_drop
    pct_node_drop = args.pct_node_drop

    weight_decay = args.weight_decay 

    trainset = Dataset(
        dataset=dataset,
        path=path,
        name=name,
        csv_file=csv_file,
        prob_edge_perturb=prob_edge_perturb,
        pct_edge_perturb=pct_edge_perturb,
        prob_node_drop=prob_node_drop,
        pct_node_drop=pct_node_drop,
        img_size=img_size,
        train=True
    )
    testset = Dataset(
        dataset=dataset,
        path=path,
        name=name,
        csv_file=csv_file,
        prob_edge_perturb=prob_edge_perturb,
        pct_edge_perturb=pct_edge_perturb,
        prob_node_drop=prob_node_drop,
        pct_node_drop=pct_node_drop,
        img_size=img_size,
        train=False
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=custom_collate_fn
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=custom_collate_fn
    )
    # network
    network = SpaCLR(
        num_node_features=1,
        last_dim=last_dim,
        hidden_dim=hidden_dim, 
        image_dims=image_dims,
        p_drop=p_drop,
        n_pos=trainset.n_pos,
        backbone='densenet', 
        projection_dims=[last_dim, last_dim]
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    # log
    save_name = f'{name}_{args.w_g2g}_{args.w_i2i}_{args.w_recon}'
    log_dir = os.path.join('log', log_name, save_name)


    trainer = TrainerSpaCLR(args, trainset.n_clusters, network, optimizer, log_dir, device=device)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None:
        trainer.load_model(checkpoint_path)
        
    trainer.fit(trainloader, testloader, epochs, dataset, name, path, checkpoint_path)
    xg, xi, _, _= trainer.valid(testloader)
    z = xg + xi * 0.1
    ari, pred_label = get_predicted_results(dataset, name, path, z)
    print("Ari value : ", ari)

    if not os.path.exists("output"):
        os.mkdir("output")
    pd.DataFrame({"cluster_labels": pred_label}).to_csv(
        "output/" + name + "_pred.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # preprocess
    parser.add_argument('--dataset', type=str, default="") # DLPFC or her2st or IDC
    parser.add_argument('--path', type=str, default="") #dataset path
    parser.add_argument('--csv_file', type=str, default="") #pathway graph information
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--last_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0007)
    parser.add_argument('--p_drop', type=float, default=0.3)
    parser.add_argument('--w_g2g', type=float, default=0.3)
    parser.add_argument('--w_i2i', type=float, default=0.2)
    parser.add_argument('--w_recon', type=float, default=0)

    # data augmentation
    parser.add_argument('--prob_edge_perturb', type=float, default=0.5)
    parser.add_argument('--pct_edge_perturb', type=float, default=0.1)
    parser.add_argument('--prob_node_drop', type=float, default=0.5)
    parser.add_argument('--pct_node_drop', type=float, default=0.1)

    # train
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--device', type=str, default="CUDA")
    parser.add_argument('--log_name', type=str, default="log_name")
    parser.add_argument('--name', type=str, default="")
    
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--checkpoint_path', type=str, default=None)

    args = parser.parse_args()
    train(args, args.name)

