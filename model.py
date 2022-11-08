import torch
import torch_geometric
import torch_geometric.data
import torch.nn.functional as F
from AU_GaphNet import AU_Net
from path_config import dir_path
from Mydataset import TrDataset, ValDataset, TestDataset
from arguements_default import default_args
from DiffusionProcess import get_edges, get_adj_matrix, get_clipped_matrix, get_top_k_matrix, get_heat_matrix



# configure the device for running the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the parameters
args = default_args()



# load primitive dataset
train_dataset = TrDataset(root=dir_path()+args.train)
train_data = train_dataset.data


# model to GPU
model = AU_Net(vae_ins=args.vae_ins, vae_hids0=args.vae_hids0, vae_lats=args.vae_lats, vae_hids1=args.vae_hids1,
               vae_outs=args.vae_outs, edgeconv_ins=args.edgeconv_ins, edgeconv_hids=args.edgeconv_hids,
               edgeconv_outs=args.edgeconv_outs, gat_ins=args.gat_ins, gat_hids=args.gat_hids, gat_outs=args.gat_outs)
model, train_dataset = model.to(device), train_data.to(device)


# define loss

loss = F.cross_entropy()


# define optimizer
optimizer = torch.optim.Adam([dict(params=model.vae_layer.parameters(), weight_decay=5e-4),
    dict(params=model.edgeconv_layer.parameters(), weight_decay=4e-5), dict(params=model.gatconv_layer.parameters(),
                                                                            weight_decay=0)], lr=args.lr)
schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# define training process



# define validation & test process
val_dataset = ValDataset(root=dir_path()+args.val)




test_dataset = TestDataset(root=dir_path()+args.test)

# visualization of training & validation & test process