from scipy.linalg import expm
from torch_geometric.data import InMemoryDataset
import numpy as np
import torch


def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.

    return adj_matrix

def get_heat_matrix(adj_matrix: np.array, t: float = 5.0) -> np.array:
    num_nodes = adj_matrix.shape[0]
    A_T = adj_matrix + np.eye(num_nodes)
    D_T = np.diag(1/np.sqrt(A_T.sum(axis=1)))
    H = D_T @ A_T @ D_T

    return expm(-t * (np.eye(num_nodes) - H))

def get_clipped_matrix(A: np.ndarray, eps: float = 0.001) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero

    return A/norm

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_edges(heat_matrix):
    edges_i = []
    edges_j = []
    edge_attr = []
    for i, row in enumerate(heat_matrix):
        for j in np.where(row > 0)[0]:
            edges_i.append(i)
            edges_j.append(j)
            edge_attr.append(heat_matrix[i, j])
    edge_index = [edges_i, edges_j]

    return torch.LongTensor(edge_index), torch.FloatTensor(edge_attr)
