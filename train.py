import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.dataset import WeatherDataset
import xarray as xr
from datetime import datetime, timedelta
from mamba import Mamba, ModelArgs

# Config
start_date_str = "2020-01-01"
end_date_str = "2020-01-01"
time_interval_hours = 6
data_dir = "data"
test_size = 0.2
batch_size = 1

model_args = ModelArgs(
    d_model=2,
    n_layer=4,
    features=5,
    d_state=126,
    expand=2,
)


def load_data_from_zarr(zarr_path, start_date_str, end_date_str, time_interval_hours):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    ds = xr.open_zarr(zarr_path)

    start_time = start_date
    end_time = end_date + timedelta(days=1) - timedelta(hours=time_interval_hours)

    ds_filtered = ds.sel(time=slice(start_time, end_time))

    surface_vars = [
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
    ]
    upper_vars = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ]

    surface_data = ds_filtered[surface_vars]
    upper_data = ds_filtered[upper_vars]

    # Extract NumPy arrays from DataArrays within the Dataset
    surface_numpy_list = [surface_data[var].values for var in surface_vars]
    upper_numpy_list = [upper_data[var].values for var in upper_vars]

    # Stack the NumPy arrays along a new dimension (channel dimension)
    surface_numpy = np.stack(surface_numpy_list, axis=0)
    upper_numpy = np.stack(upper_numpy_list, axis=0)

    surface_tensor = torch.from_numpy(surface_numpy).float().transpose(0, 1)
    upper_tensor = torch.from_numpy(upper_numpy).float().transpose(0, 1)

    times = ds_filtered.time.values

    return surface_tensor, upper_tensor, times


# Example usage:
zarr_path = "output.zarr"  # Replace with your Zarr file path
start_date_str = "2020-01-01"
end_date_str = "2020-01-02"
time_interval_hours = 6  # or other interval

surface_tensor, upper_tensor, times = load_data_from_zarr(
    zarr_path, start_date_str, end_date_str, time_interval_hours
)

print("Surface Tensor shape:", surface_tensor.shape)
print("Upper Tensor shape:", upper_tensor.shape)
print("times array shape:", times.shape)

# surface_tensor = torch.from_numpy(surface_data)
# upper_tensor = torch.from_numpy(upper_data)

dataset = WeatherDataset(surface_tensor, upper_tensor)
print(len(dataset))
dataloader = DataLoader(dataset)

inputs, targets = dataset[0]
surface, upper = inputs

model = Mamba(model_args)

# upper = torch.unsqueeze(upper, 0)
# output = model(upper)
# print(output.shape)

# Split into test and train datasets
test_size = int(len(dataset) * test_size)
train_size = len(dataset) - test_size

indices = np.arange(len(dataset))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)

for i in train_dataset:
    print(len(i))
