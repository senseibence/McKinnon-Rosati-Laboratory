import os
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

input_path = "../arrays"
output_path = "/gpfs/scratch/blukacsy/granulomas_final_torch_nn_v1.pt"

hidden_layers = [8192, 4096, 2048, 1024]
dropout_rate = 0.5
l2_reg = 1e-4
learning_rate = 1e-5
epochs = 1000
batch_size = 512

class NumpyArrayDataset(Dataset):
    def __init__(self, features, labels, sample_weights=None):
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.sample_weights is not None:
            w = self.sample_weights[idx]
            return x, y, w
        else:
            return x, y

class SimpleFeedForwardNN(nn.Module):
    
    def __init__(self, input_size, num_classes, hidden_layers, dropout_rate, l2_reg=1e-4):
        super(SimpleFeedForwardNN, self).__init__()

        layers = []

        layers.append(nn.BatchNorm1d(input_size))

        in_dim = input_size
        for h in hidden_layers:
            
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            
            
            layers.append(nn.BatchNorm1d(h))
            
            layers.append(nn.Dropout(dropout_rate))

            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

def ddp_setup(rank: int, world_size: int):
    """
    Initialize the default process group.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)
    
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_one_epoch(model, dataloader, optimizer, device, epoch, world_size):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        if len(batch) == 3:
            x, y, w = batch
            w = w.to(device)
        else:
            x, y = batch
            w = None

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits, y)

        if w is not None:
            losses = losses * w

        loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, w = batch
                w = w.to(device)
            else:
                x, y = batch
                w = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            losses = loss_fn(logits, y)
            if w is not None:
                losses = losses * w

            loss = losses.mean()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def main(rank, world_size):
    ddp_setup(rank, world_size)

    torch.manual_seed(42)
    np.random.seed(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    train_features = np.load(os.path.join(input_path, "train_features.npy"))
    val_features = np.load(os.path.join(input_path, "val_features.npy"))
    train_labels = np.load(os.path.join(input_path, "train_labels.npy"))
    val_labels = np.load(os.path.join(input_path, "val_labels.npy"))
    sample_weights = np.load(os.path.join(input_path, "sample_weights.npy"))
    
    train_features = torch.from_numpy(train_features).float()
    val_features = torch.from_numpy(val_features).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_labels = torch.from_numpy(val_labels).long()
    
    if sample_weights is not None:
        sample_weights = torch.from_numpy(sample_weights).float()

    train_dataset = NumpyArrayDataset(train_features, train_labels, sample_weights)
    val_dataset   = NumpyArrayDataset(val_features, val_labels)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    input_size  = train_features.shape[1]  # 23693
    num_classes = len(torch.unique(train_labels))  # 30

    model = SimpleFeedForwardNN(
        input_size     = input_size,
        num_classes    = num_classes,
        hidden_layers  = hidden_layers,
        dropout_rate   = dropout_rate,
        l2_reg         = l2_reg
    )

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.to(device)

    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    for epoch in range(epochs):
       
        train_sampler.set_epoch(epoch)

        
        train_loss, train_acc = train_one_epoch(ddp_model, train_loader, optimizer, device, epoch, world_size)

        
        val_loss, val_acc = validate(ddp_model, val_loader, device)

        if rank == 0:
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if rank == 0:
        # ddp_model.module.state_dict() is the underlying model
        torch.save(ddp_model.module.state_dict(), output_path)
        print(f"Model saved to {output_path}")

    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPU found. DDP requires at least one GPU.")
    
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )
