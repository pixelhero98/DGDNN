import torch
import torch.nn.functional as F
from dataset_gen import Mydataset
from dgdnn import DGDNN
from torch_geometric.logging import log
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from matplotlib import axes
import seaborn as sns
import sklearn.preprocessing as skp
from sklearn.metrics import matthews_corrcoef, f1_score

# Configure the device for running the model on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Configure the default variables // # these can be tuned // # examples
fast_approx = False # True for fast approximation and implementation
sedate = ['2013-01-01', '2014-12-31']  # these can be tuned
val_sedate = ['2015-01-01', '2015-06-30'] # these can be tuned
test_sedate = ['2015-07-01', '2017-12-31'] # these can be tuned
market = ['NASDAQ', 'NYSE', 'SSE'] # can be changed
dataset_type = ['Train', 'Validation', 'Test']
com_path = ['/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NASDAQ.csv',
            '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NYSE.csv',
            '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/NYSE_missing.csv']
des = '/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/raw_stock_data/stocks_indicators/data'
directory = "/content/drive/MyDrive/Raw_Data/Stock_Markets/NYSE_NASDAQ/raw_stock_data/stocks_indicators/data/google_finance"
NASDAQ_com_list = []
NYSE_com_list = []
NYSE_missing_list = []
com_list = [NASDAQ_com_list, NYSE_com_list, NYSE_missing_list]
for idx, path in enumerate(com_path):
    with open(path) as f:
        file = csv.reader(f)
        for line in file:
            com_list[idx].append(line[0])  # append first element of line if each line is a list
NYSE_com_list = [com for com in NYSE_com_list if com not in NYSE_missing_list]


# Generate datasets
train_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, sedate[0], sedate[1], 19, dataset_type[0], fast_approx)
validation_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, val_sedate[0], val_sedate[1], 19, dataset_type[1], fast_approx)
test_dataset = MyDataset(directory, des, market[0], NASDAQ_com_list, test_sedate[0], test_sedate[1], 19, dataset_type[2], fast_approx)


# Define model
layers, num_nodes, expansion_step, num_heads, active, timestamp, classes = 6, 1026, 7, 2, [True, True, True, True, True, True], 19, 2
diffusion_size = [95, 64, 128, 256, 256, 256, 128]
emb_size = [64 + 64, 128 + 256, 256 + 256, 256 + 256, 256 + 256, 128 + 256]  
emb_hidden_size, emb_output_size, raw_feature_size = 1024, 256, 64
model = DGDNN(diffusion_size, emb_size, emb_hidden_size, emb_output_size, raw_feature_size, classes, layers, num_nodes, expansion_step, num_heads, active).to(device)

# Pass model GPU
model = model.to(device)


# Define optimizer and objective function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.5e-5)

# def neighbor_distance_regularizer(theta):
#     box = torch.sum(theta, dim=-1)
#     result = torch.zeros_like(theta)

#     for idx, row in enumerate(theta):
#         for i, j in enumerate(row):
#             result[idx, i] = i * j

#     result_sum = torch.sum(result, dim=1)
#     return torch.sum(result / result_sum[:, None])


# Define training process & validation process & testing process
epochs = 6000

# Training
for epoch in range(epochs):
    model.train()
    objective_total = 0
    correct = 0
    total = 0

    for sample in train_dataset: #Recommend to update every sample, full batch training can be time-consuming
        X = sample['X'].to(device)  # node feature tensor
        A = sample['A'].to(device)  # adjacency tensor
        C = sample['Y'].long()
        C = C.to(device)  # label vector
        optimizer.zero_grad()
        out = model(X, A)
        objective = F.cross_entropy(out, C) # for efficiency can omit the regularization term - 0.0029 * neighbor_distance_regularizer(model.theta)
        objective.backward()
        optimizer.step()
        objective_total += objective.item()

    # If performance progress of the model is required
        out = out.argmax(dim=1)
        correct += int((out == C).sum())
        total += C.shape[0]
    if epoch % 1 == 0:
            print(f"Epoch {epoch}: train_loss={objective_total:.4f}, train_acc={correct / total:.4f}")

# Validation
model.eval()

# Define evaluation metrics
# ACC, MCC, and F1
acc = 0
f1 = 0
mcc = 0

for idx, sample in enumerate(validation_dataset):

    X = sample['X']  # node feature tensor
    A = sample['A']  # adjacency tensor
    C = sample['Y']  # label vector
    out = model(X, A).argmax(dim=1)

    acc += int((out == C).sum())
    f1 += f1_score(C, out.cpu().numpy())
    mcc += matthews_corrcoef(C, out.cpu().numpy())

print(acc / (len(validation_dataset) * C.shape[0]))
print(f1 / len(validation_dataset))
print(mcc/ len(validation_dataset))

# Test

acc = 0
f1 = 0
mcc = 0

for idx, sample in enumerate(test_dataset):
    X = sample['X']  # node feature tensor
    A = sample['A']  # adjacency tensor
    C = sample['Y']  # label vector
    out = model(X, A).argmax(dim=1)

    acc += int((out == C).sum())
    f1 += f1_score(C, out.cpu().numpy())
    mcc += matthews_corrcoef(C, out.cpu().numpy())

print(acc / (len(test_dataset) * C.shape[0]))
print(f1 / len(test_dataset))
print(mcc / len(test_dataset))
