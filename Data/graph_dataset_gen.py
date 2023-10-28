from scipy.linalg import expm
import torch
import csv
import os
import numpy as np
import random
import sklearn.preprocessing as skp
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class MyDataset(Dataset):
  def __init__(self, root: str, desti: str, market: str, comlist: List[str], start: str, end: str, window: int, dataset_type: str):

      super().__init__()

      self.comlist = comlist
      self.market = market
      self.root = root
      self.desti = desti
      self.start = start
      self.end = end
      self.window = window
      self.dates, self.next_day = self.find_dates(self.start, self.end, self.root, self.comlist, self.market)
      self.dataset_type = dataset_type
      # Check if graph files already exist
      graph_files_exist = all(os.path.exists(os.path.join(self.desti, f'{self.market}_{self.dataset_type}_{self.start}_{self.end}_{self.window}/graph_{i}.pt')) for i in range(len(self.dates) - self.window + 1))
      #print('1')
      if not graph_files_exist:
          # Generate the graphs and save them as PyTorch tensors
          self._create_graphs(self.dates, self.desti, self.comlist, self.market, self.root, self.window, self.next_day)
          #print('1')

  def __len__(self):
    return len(self.dates - self.window + 1)

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

    # Get the dates for each company in the target list
    date_sets = []
    after_end_date_sets = []
    for h in comlist:
      dates = set()
      after_end_dates = set()
      d_path = path + '/' + market + '_' + h + '_30Y.csv'
      with open(d_path) as f:
        file = csv.reader(f)
        next(file, None)  # Skip the header row
        for line in file:
          date_str = line[0][:10]
          if self.check_years(date_str, start, end):
            dates.add(date_str)
          elif self.check_years(date_str, end, '2017-12-31'):  # Assuming there won't be any dates later than '2018-12-31'
            after_end_dates.add(date_str)

      date_sets.append(dates)
      after_end_date_sets.append(after_end_dates)

    # Find the intersection of all date sets
    all_dates = list(set.intersection(*date_sets))
    all_after_end_dates = list(set.intersection(*after_end_date_sets))

    # Find the next common day after the end date
    next_common_day = min(all_after_end_dates) if all_after_end_dates else None

    return sorted(all_dates), next_common_day



  @lru_cache(maxsize=None)
  def signal_energy(self, x_tuple: Tuple[float]) -> float:
    x = np.array(x_tuple)

    return np.sum(np.square(x))

  @lru_cache(maxsize=None)
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
        if entropy[i] < entropy[j]:
          A[i, j] = torch.tensor(energy[i] / energy[j], dtype=torch.float32)
        else:
          A[i, j] = 0.

    return A


  def node_feature_matrix(self, dates: List[str], comlist: List[str], market: str, path: str) -> torch.Tensor:
    # Convert dates to datetime format for easier comparison
    dates_dt = [pd.to_datetime(date).date() for date in dates]

    # Initialize the tensor
    X = torch.zeros((5, len(comlist), len(dates_dt)))

    for idx, h in enumerate(comlist):
      d_path = path + '/' + market + '_' + h + '_30Y.csv'

      # Read the entire CSV file into a DataFrame, but only the rows with date in 'dates_dt'
      df = pd.read_csv(d_path, parse_dates=[0], index_col=0)
      df.index = df.index.astype(str).str.split(" ").str[0]
      df.index = pd.to_datetime(df.index)
      df = df[df.index.isin(dates_dt)]

      # Transpose the DataFrame so that the dates become columns and the fields become rows
      df_T = df.transpose()

      # Select only the rows for the fields we're interested in, which are the ones with indices 1 to 5
      df_selected = df_T.iloc[0:5]

      # Convert the selected part of the DataFrame to a numpy array and assign it to the tensor
      X[:, idx, :] = torch.from_numpy(df_selected.to_numpy())

    return X


  def _create_graphs(self, dates: List[str], desti: str, comlist: List[str], market: str, root: str, window: int, next_day: str):

    dates.append(next_day)

    # Wrap the range function with tqdm to create a progress bar
    for i in tqdm(range(len(dates) - window)):

        # Construct the filename for the current graph
        directory_path = os.path.join(desti, f'{market}_{self.dataset_type}_{self.start}_{self.end}_{window}')
        filename = os.path.join(directory_path, f'graph_{i}.pt')

        # If the file already exists, skip to the next iteration
        if os.path.exists(filename):
            print(f"Graph {i + 1}/{len(dates) - window} already exists, skipping...")
            continue

        print(f'Generating graph {i + 1}/{len(dates) - window}...')

        # Generate labels
        box = dates[i:i + window + 1]
        X = self.node_feature_matrix(box, comlist, market, root)
        C = torch.zeros(X.shape[1])
        for j in range(C.shape[0]):
          if X[3, j, -1] - X[3, j, -2] > 0:
            C[j] = 1

        # Slice the desired data and do normalization on raw data
        X = X[:,:,:-1]
        for i in range(X.shape[0]):
          # Adding 1 before taking log to avoid log(0)
          X[i] = torch.Tensor(np.log1p(X[i].numpy()))

        # Obtain adjacency tensor
        A = torch.zeros((X.shape[0], X.shape[1], X.shape[1]))
        for j in range(A.shape[0]):
            A[j] = self.adjacency_matrix(X[j])

        # Save the X, A, C tensors
        os.makedirs(directory_path, exist_ok=True)

        torch.save({'X': X, 'A': A, 'Y': C}, filename)
