import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import pandas as pd
from tqdm import tqdm
from mamba import Mamba, ModelArgs
import wandb

wandb.init(
    project="Mamba Weather Timeseries"
)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs = batch.to(device)
        targets = inputs  # For simplicity, assuming auto-regressive generation

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss


model_args = ModelArgs(d_model=512, n_layer=3, features=8)
model = Mamba(model_args)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length]
        return x, y


df = pd.read_parquet(r"data\gfs_dataframe_train_test.parquet")

columns_to_remove = ['coord_index', 'lat', 'lon']
data = torch.tensor([])
for i in sorted(list(set(list(df['lat'])))):
    for j in sorted(list(set(list(df['lon'])))):
        df_split = df[df['lat'] == i]
        df_split = df_split[df_split['lon'] == j]
        df_split = df_split.drop(columns=columns_to_remove, axis=1)
        features = torch.tensor([df_split.values]).float()
        # features = torch.tensor([df_split.values])
        data = torch.cat((data, features), dim=0)

# Example criterion:
criterion = nn.MSELoss()
# Example optimizer:
optimizer = Adam(model.parameters(), lr=0.001)

# Move the model to the desired device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()

data_split = 160
num_epochs = 1
train_pbar = tqdm(total = data.size(1) * data_split * num_epochs, desc="Epoch 1 - Training loss: None")

# torch.set_grad_enabled(True)

for epoch in range(num_epochs):
    running_loss = 0
    for i in range(data_split):
        train_dataset = TimeSeriesDataset(data[i], (7*4))
        train_loader = DataLoader(train_dataset, 
                                  sampler=SequentialSampler(train_dataset)
                                  )
        for index, batch in enumerate(train_loader):
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)[:, -1]
            # loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            train_pbar.set_description(f"Epoch {epoch+1} - Training loss: {running_loss/((i+1) * (index+1)):.5f}")
            train_pbar.update(1)

            wandb.log({
                "average training loss": (running_loss/((i+1)*(index+1))), 
                "training loss": (loss.item()),
                })


test_pbar = tqdm(total=(data.size(1) * (200 - data_split)), desc="Testing loss: None" )
running_loss = 0
for i in range(200-data_split):
    test_dataset = TimeSeriesDataset(data[data_split+i], (7*4))
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    for index, batch in enumerate(train_loader):
        inputs, targets = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(inputs)[:, -1]
        
        # loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss = criterion(outputs, targets)

        running_loss += loss.item()

        test_pbar.set_description(f"Testing loss: {running_loss/((i+1) * (index+1)):.5f}")
        test_pbar.update(1)

        wandb.log({
            "Average testing loss": running_loss/((i+1) * (index+1)),
            "Test loss": loss.item(),
        })
