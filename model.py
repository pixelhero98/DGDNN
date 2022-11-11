import torch
import torch_geometric
import torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
from AU_GaphNet import AU_Net
from path_config import dir_path
from Mydataset import TrDataset, ValDataset, TestDataset
from Graph_VAE import get_edge_feature
from arguements_default import default_args
from DiffusionProcess import get_adj_matrix, get_top_k_matrix, get_heat_matrix, get_edges
from torch_geometric.logging import log


# configure the device for running the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the parameters
args = default_args()



# load primitive dataset
train_dataset = TrDataset(root=dir_path()+args.train, transform=T.NormalizeFeatures()).data



# model to GPU
model = AU_Net(vae_ins=args.vae_ins, vae_hids0=args.vae_hids0, vae_lats=args.vae_lats, vae_hids1=args.vae_hids1,
               vae_outs=args.vae_outs, edgeconv_ins=args.edgeconv_ins, edgeconv_hids=args.edgeconv_hids,
               edgeconv_outs=args.edgeconv_outs, gat_ins=args.gat_ins, gat_hids=args.gat_hids, gat_outs=args.gat_outs,
               num_labels=args.num_labels)

model, train_dataset = model.to(device), train_dataset.to(device)


# define loss



# define optimizer
optimizer = torch.optim.Adam([dict(params=model.vae_layer.parameters(), weight_decay=5e-4),
    dict(params=model.edgeconv_layer.parameters(), weight_decay=4e-5), dict(params=model.gatconv_layer.parameters(),
                                                                            weight_decay=0)], lr=args.lr)
# schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# define training process
new_train_data = train_dataset

for epoch in range(0, args.epochs + 1):

    temp_adj = new_train_data.edge_index

    model.train()
    optimizer.zero_grad()

    new_adj = get_top_k_matrix(get_heat_matrix(get_adj_matrix(new_train_data), args.t), args.top_k)
    new_edge_index, new_edge_attr = get_edges(new_adj)

    concat_x = get_edge_feature(new_train_data.x, new_adj)

    out, x_tilda = model(new_train_data.x, new_edge_index, new_edge_attr, concat_x)
    new_train_data = Data(x=x_tilda, edge_index=new_edge_index, edge_attr=new_edge_attr)

    loss = F.cross_entropy(out, train_dataset.y) \
           + mu1 * torch.linalg.norm(get_adj_matrix(new_train_data), dim=1, ord=1) \
           + mu2 * torch.linalg.norm(get_adj_matrix(new_train_data), dim=1, ord=2) + AU_Net.vae_layer.encoder.kl
               # + lamda * torch.norm((new_train_data.edge_index - temp_adj), p='fro', dim=1) \
    loss.backward()
    optimizer.step()

    log(Epoch=epoch, Loss=loss)
    #Train=train_acc, Val=val_acc, Best_Val=best_val_acc, Test=tmp_test_acc, Best_Test=final_test_acc)

# model.save()
# define validation & test process
# val_dataset = ValDataset(root=dir_path()+args.val, transform=T.NormalizeFeatures())




# test_dataset = TestDataset(root=dir_path()+args.test,  transform=T.NormalizeFeatures())

# visualization of training & validation & test process