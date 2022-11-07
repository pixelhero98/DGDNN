import torch
import torch_geometric
import torch_geometric.data
from AU_GaphNet import AU_Net
from Mydataset import MyOwnDataset
from arguements_default import default_args
from raw_data import edge_info_generation, node_feature_generation
from DiffusionProcess import get_edges, get_adj_matrix, get_clipped_matrix, get_top_k_matrix, get_heat_matrix



# configure the device for running the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the parameters
args = default_args()



# load primitive dataset






# define loss





# define training process



# define validation & test process





# visualization of training & validation & test process