import torch
import numpy as np
from utils.dataset import WeatherDataset
import os
from datetime import datetime, timedelta
from mamba import Mamba, ModelArgs

# Config
start_date_str = "2020-01-01"
end_date_str = "2020-01-01"
time_interval_hours = 6
data_dir = "data"

model_args = ModelArgs(
    d_model=2,
    n_layer=4,
    features=5,
    d_state=126,
    expand=2,
)


# Load data
def load_data(data_dir, date_str, time_str):
    dir_path = os.path.join(data_dir, date_str, time_str)

    try:
        input_surface = np.load(os.path.join(dir_path, "input_surface.npy"))
        input_upper = np.load(os.path.join(dir_path, "input_upper.npy"))
        return input_surface, input_upper
    except FileNotFoundError:
        print(f"Error: Data not found in {dir_path}")
        return None, None


start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
current_date = start_date

surface_data_list = []
upper_data_list = []
times_list = []

while current_date <= end_date:
    for hour in range(0, 24, time_interval_hours):
        time_str = f"{hour:02d}:00"
        date_str = current_date.strftime("%Y-%m-%d")

        input_surface, input_upper = load_data(data_dir, date_str, time_str)

        if input_surface is not None and input_upper is not None:
            surface_data_list.append(input_surface)
            upper_data_list.append(input_upper)
            times_list.append(
                datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            )
    current_date += timedelta(days=1)

surface_data = np.stack(surface_data_list, axis=0)
upper_data = np.stack(upper_data_list, axis=0)

surface_tensor = torch.from_numpy(surface_data)
upper_tensor = torch.from_numpy(upper_data)

dataset = WeatherDataset(surface_tensor, upper_tensor)

inputs, targets = dataset[0]
surface, upper = inputs
# num_features = 5
# num_enc_features = 1
# patch_size = (2, 4, 4)
# batch_size = 1

# pos_encoding = PositionalEncodingPermute3D(num_enc_features)
# conv = torch.nn.Conv3d(num_features, num_enc_features, patch_size, patch_size)

model = Mamba(model_args)

upper = torch.unsqueeze(upper, 0)
output = model(upper)
print(output.shape)
