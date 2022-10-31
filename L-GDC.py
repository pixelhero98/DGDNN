import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import inspect
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.04)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args(args=[])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = osp.join(osp.dirname(osp.abspath(filename)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='coeff', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not args.use_gdc)
        self.linblock = torch.nn.Sequential(torch.nn.Linear(hidden_channels, 128), torch.nn.ReLU(),
                                            torch.nn.Linear(128, hidden_channels),
                                            torch.nn.ReLU())

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.linblock(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = final_test_acc = 0
epoch_l = []
loss_l = []
train_l = []
val_l = []
test_l = []
for epoch in range(1, args.epochs + 1):
    loss = train()
    if epoch < 30:
      scheduler.step()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    if tmp_test_acc > final_test_acc:
        final_test_acc = tmp_test_acc
    epoch_l.append(epoch)
    loss_l.append(loss)
    train_l.append(train_acc)
    val_l.append(val_acc)
    test_l.append(tmp_test_acc)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Best_Val=best_val_acc, Test=tmp_test_acc, Best_Test=final_test_acc)

plt.plot(epoch_l, loss_l, label = 'loss') 
plt.plot(epoch_l, train_l, label = 'train_acc')
plt.plot(epoch_l, val_l, label = 'val_acc')
plt.plot(epoch_l, test_l, label = 'test_acc')
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('values')
plt.savefig("/data/h_gdc_CORA.png")
