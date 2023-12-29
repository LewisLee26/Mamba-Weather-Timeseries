import pandas as pd

# Read Parquet file into a DataFrame
df = pd.read_parquet('gfs_dataframe_1.parquet')

# Display the DataFrame
print(df)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from mamba import Mamba, ModelArgs

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

# Example usage:
# Define your ModelArgs and create an instance of the Mamba model
model_args = ModelArgs(d_model=512, n_layer=1, vocab_size=8)
mamba_model = Mamba(model_args)

# Define your data loader, criterion, and optimizer
# Note: You need to replace DataLoader, criterion, and optimizer with your actual data loader, criterion, and optimizer
# Example DataLoader:
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Example criterion:
criterion = nn.MSELoss()
# Example optimizer:
optimizer = Adam(mamba_model.parameters(), lr=0.001)

# Move the model to the desired device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mamba_model.to(device)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     average_loss = train_model(mamba_model, train_loader, criterion, optimizer, device)
#     print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")