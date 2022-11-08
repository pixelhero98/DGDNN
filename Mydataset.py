import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
from Cross_Attention_Score import cross_attention_score
from raw_data import edge_info_generation, node_feature_label_generation
from path_config import dataset_seg


class MyOwnDataset(InMemoryDataset):
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

        start, end = dataset_seg(index='test')
        X1 = []
        X1, labels = node_feature_generation(X1, start, end)
        Edge_index, Edge_attr = edge_info_generation(X1)
        Edge_attr = Edge_attr.reshape(Edge_attr.shape[0], 1)
        data = Data(x=X1.type(torch.float), edge_index=Edge_index.type(torch.long),
                    edge_attr=Edge_attr, y=labels.type(torch.float))

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

