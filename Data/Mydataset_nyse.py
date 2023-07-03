import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
from raw_data_nyse import edge_info_generation, node_feature_label_generation, mask_generation
import sklearn.preprocessing as skp

class TrDataset_0_nyse(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        start = 2016
        end = 2016
        X1 = []
        X1, labels = node_feature_label_generation(X1, start, end)
        Edge_index, Edge_attr = edge_info_generation(X1, 'heat', 'threshold')
        tr, val, test = mask_generation(X1)
        data = Data(x=torch.Tensor(skp.normalize(X1, axis=0)).type(torch.float), edge_index=Edge_index.type(torch.long),
                    edge_attr=torch.Tensor(skp.normalize(Edge_attr, axis=0)).type(torch.float),
                    y=labels.type(torch.int64), train_mask=tr,
                    val_mask=val, test_mask=test)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class TrDataset_1_nyse(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        start = 2017
        end = 2017
        X1 = []
        X1, labels = node_feature_label_generation(X1, start, end)
        Edge_index, Edge_attr = edge_info_generation(X1, 'heat', 'threshold')
        tr, val, test = mask_generation(X1)
        data = Data(x=torch.Tensor(skp.normalize(X1, axis=0)).type(torch.float), edge_index=Edge_index.type(torch.long),
                    edge_attr=torch.Tensor(skp.normalize(Edge_attr, axis=0)).type(torch.float),
                    y=labels.type(torch.int64), train_mask=tr,
                    val_mask=val, test_mask=test)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class TrDataset_2_nyse(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        start = 2015
        end = 2015
        X1 = []
        X1, labels = node_feature_label_generation(X1, start, end)
        Edge_index, Edge_attr = edge_info_generation(X1, 'heat', 'threshold')
        tr, val, test = mask_generation(X1)
        data = Data(x=torch.Tensor(skp.normalize(X1, axis=0)).type(torch.float), edge_index=Edge_index.type(torch.long),
                    edge_attr=torch.Tensor(skp.normalize(Edge_attr, axis=0)).type(torch.float),
                    y=labels.type(torch.int64), train_mask=tr,
                    val_mask=val, test_mask=test)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
