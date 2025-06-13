import torch
import math
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.linalg import expm


class MyDataset(Dataset):
    def __init__(self, root: str, desti: str, market: str, comlist: List[str], start: str, end: str, window: int, dataset_type: str, fast_approx):
        super().__init()

        self.comlist = comlist
        self.market = market
        self.root = root
        self.desti = desti
        self.start = start
        self.end = end
        self.window = window
        self.dates, self.next_day = self.find_dates(start, end, root, comlist, market)
        self.dataset_type = dataset_type
        self.fast_approx = fast_approx
        # Check if graph files already exist
        graph_files_exist = all(os.path.exists(os.path.join(desti, f'{market}_{dataset_type}_{start}_{end}_{window}/graph_{i}.pt')) for i in range(len(self.dates) - window + 1))

        if not graph_files_exist:
            # Generate the graphs and save them as PyTorch tensors
            self._create_graphs(self.dates, desti, comlist, market, root, window)

    def __len__(self):
        
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int):
        directory_path = os.path.join(self.desti, f'{self.market}_{self.dataset_type}_{self.start}_{self.end}_{self.window}')
        data_path = os.path.join(directory_path, f'graph_{idx}.pt')
        if os.path.exists(data_path):
            
            return torch.load(data_path)
        else:
            
            raise FileNotFoundError(f"No graph data found for index {idx}")

    def check_years(self, date_str: str, start_str: str, end_str: str) -> bool:
        date_format = "%Y-%m-%d"
        date = datetime.strptime(date_str, date_format)
        start = datetime.strptime(start_str, date_format)
        end = datetime.strptime(end_str, date_format)
        
        return start <= date <= end

    def find_dates(self, start: str, end: str, path: str, comlist: List[str], market: str) -> Tuple[List[str], str]:
        date_sets = []
        after_end_date_sets = []

        for h in comlist:
            dates = set()
            after_end_dates = set()
            d_path = os.path.join(path, f'{market}_{h}_30Y.csv')

            with open(d_path) as f:
                file = csv.reader(f)
                next(file, None)  # Skip the header row
                for line in file:
                    date_str = line[0][:10]

                    if self.check_years(date_str, start, end):
                        dates.add(date_str)
                    elif self.check_years(date_str, end, '2017-12-31'):  # '2017-12-31' is just an example, replacing with the latest date when datasets change
                        after_end_dates.add(date_str)

            date_sets.append(dates)
            after_end_date_sets.append(after_end_dates)

        all_dates = list(set.intersection(*date_sets))
        all_after_end_dates = list(set.intersection(*after_end_date_sets))
        next_common_day = min(all_after_end_dates) if all_after_end_dates else None

        return sorted(all_dates), next_common_day

    def signal_energy(self, x_tuple: Tuple[float]) -> float:
        x = np.array(x_tuple)
        
        return np.sum(np.square(x))

    def information_entropy(self, x_tuple: Tuple[float]) -> float:
        x = np.array(x_tuple)
        unique, counts = np.unique(x, return_counts=True)
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        return entropy

    def adjacency_matrix(self, X: torch.Tensor) -> torch.Tensor:
        A = torch.zeros((X.shape[0], X.shape[0]))
        X = X.numpy()
        energy = np.array([self.signal_energy(tuple(x)) for x in X])
        entropy = np.array([self.information_entropy(tuple(x)) for x in X])

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                concat_x = np.concatenate((X[i], X[j]))
                A[i, j] = torch.tensor((energy[i] / energy[j]) * (math.exp(entropy[i] + entropy[j] - self.information_entropy(tuple(concat_x)))), dtype=torch.float32)
                
        if self.fast_approx:
            t = 5
            A = A.numpy()
            num_nodes = A.shape[0]
            A_tilde = A + np.eye(num_nodes)
            D_tilde = np.diag(1 / np.sqrt(A.sum(axis=1)))
            H = D_tilde @ A @ D_tilde
            return expm(-t * (np.eye(num_nodes) - H))
            
        A[A<1] = 1
        
        return torch.log(A)

    def node_feature_matrix(self, dates: List[str], comlist: List[str], market: str, path: str) -> torch.Tensor:
        dates_dt = [pd.to_datetime(date).date() for date in dates]
        X = torch.zeros((5, len(comlist), len(dates_dt)))

        for idx, h in enumerate(comlist):
            d_path = os.path.join(path, f'{market}_{h}_30Y.csv')
            df = pd.read_csv(d_path, parse_dates=[0], index_col=0)
            df.index = df.index.astype(str).str.split(" ").str[0]
            df.index = pd.to_datetime(df.index)
            df = df[df.index.isin(dates_dt)]
            df_T = df.transpose()
            df_selected = df_T.iloc[0:5]
            X[:, idx, :] = torch.from_numpy(df_selected.to_numpy())

        return X

    def _create_graphs(self, dates: List[str], desti: str, comlist: List[str], market: str, root: str, window: int):
            dates.append(self.next_day)
    
            for i in tqdm(range(len(dates) - window + 1)):
                directory_path = os.path.join(desti, f'{market}_{self.dataset_type}_{self.start}_{self.end}_{window}')
                filename = os.path.join(directory_path, f'graph_{i}.pt')
    
                if os.path.exists(filename):
                    print(f"Graph {i}/{len(dates) - window + 1} already exists, skipping...")
                    continue
    
                print(f'Generating graph {i}/{len(dates) - window + 1}...')
    
                box = dates[i:i + window + 1]
                X = self.node_feature_matrix(box, comlist, market, root)
                C = torch.zeros(X.shape[1])
    
                for j in range(C.shape[0]):
                    if X[3, j, -1] - X[3, j, -2] > 0:
                        C[j] = 1
    
                X = X[:, :, :-1]
                X_dim = [X.shape[0], X.shape[-1]]
                X = X.view(-1, X_dim[-1])
                X = torch.chunk(X, X_dim[0], dim=0)
                X = torch.cat(X, dim=1)
                X = torch.Tensor(np.log1p(X.numpy()))
                A = self.adjacency_matrix(X)
                C = C.long()
    
                os.makedirs(directory_path, exist_ok=True)
    
                torch.save({'X': X, 'A': A, 'Y': C}, filename)

