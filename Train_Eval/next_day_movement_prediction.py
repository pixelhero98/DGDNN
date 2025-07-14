import csv
import torch
import torch.nn.functional as F
from dataset_gen import MyDataset
from dgdnn import DGDNN
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

# Configure the device for running the model on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration (tunable)
fast_approx = False
sedate       = ['2013-01-01', '2014-12-31']
val_sedate   = ['2015-01-01', '2015-06-30']
test_sedate  = ['2015-07-01', '2017-12-31']
market       = ['NASDAQ', 'NYSE', 'SSE']
dataset_type = ['train', 'validation', 'test']

# Paths to company lists
com_path = [
    '/content/drive/MyDrive/.../NASDAQ.csv',
    '/content/drive/MyDrive/.../NYSE.csv',
    '/content/drive/MyDrive/.../NYSE_missing.csv',
]

des = '/content/.../data'
directory = '/content/.../google_finance'

# Read tickers
NASDAQ_com_list = []
NYSE_com_list  = []
NYSE_missing   = [] # replace with the missing tickers 
com_lists = [NASDAQ_com_list, NYSE_com_list, NYSE_missing]
for idx, path in enumerate(com_path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                com_lists[idx].append(row[0])
# filter out missing
NYSE_com_list = [c for c in NYSE_com_list if c not in NYSE_missing]

# Instantiate datasets
train_dataset = MyDataset(
    root=directory,
    dest=des,
    market=market[0],
    tickers=NASDAQ_com_list,
    start=sedate[0],
    end=sedate[1],
    window=19,
    mode=dataset_type[0],
    fast_approx=fast_approx
)
validation_dataset = MyDataset(
    root=directory,
    dest=des,
    market=market[0],
    tickers=NASDAQ_com_list,
    start=val_sedate[0],
    end=val_sedate[1],
    window=19,
    mode=dataset_type[1],
    fast_approx=fast_approx
)
test_dataset = MyDataset(
    root=directory,
    dest=des,
    market=market[0],
    tickers=NASDAQ_com_list,
    start=test_sedate[0],
    end=test_sedate[1],
    window=19,
    mode=dataset_type[2],
    fast_approx=fast_approx
)

# Model hyperparameters
layers, num_nodes, expansion_step, num_heads = 6, len(NASDAQ_com_list), 7, 2
classes = 2
emb_hidden_size, emb_output_size, raw_feature_size = 1024, 256, 64
timestamp = 19

diffusion_size = [timestamp * 5, 64, 128, 256, 256, 256, 128]
emb_size       = [64+64, 128+256, 256+256, 256+256, 256+256, 128+256]
if num_heads != 2:
    scale = num_heads / 2.0
    emb_output_size  = int(round(emb_output_size  * scale))
    raw_feature_size = int(round(raw_feature_size * scale))
    diffusion_size = [diffusion_size[0]] + [int(round(x * scale)) for x in diffusion_size[1:]]
    emb_size       = [int(round(x * scale)) for x in emb_size]

# Initialize model
model = DGDNN(
    diffusion_size,
    emb_size,
    emb_hidden_size,
    emb_output_size,
    raw_feature_size,
    classes,
    layers,
    num_nodes,
    expansion_step,
    num_heads,
    active=[True]*layers
).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.5e-5)

epochs = 6000

# Training loop
def train():
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for sample in train_dataset:
            X = sample['X'].to(device)
            A = sample['A'].to(device)
            C = sample['Y'].to(device).long()
            optimizer.zero_grad()
            logits = model(X, A)
            loss = F.cross_entropy(logits, C)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += int((preds == C).sum())
            total += C.size(0)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={total_loss:.4f}, acc={correct/total:.4f}")

# Evaluation function
def evaluate(dataset, name="Val"):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for sample in dataset:
            X = sample['X'].to(device)
            A = sample['A'].to(device)
            C = sample['Y'].to(device)
            logits = model(X, A)
            preds = logits.argmax(dim=1).cpu().numpy()
            trues = C.cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(trues)
    acc = accuracy_score(all_trues, all_preds)
    f1  = f1_score(all_trues, all_preds, average='macro')
    mcc = matthews_corrcoef(all_trues, all_preds)
    print(f"{name} -- Acc: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

if __name__ == "__main__":
    train()
    evaluate(validation_dataset, name="Validation")
    evaluate(test_dataset,       name="Test")
