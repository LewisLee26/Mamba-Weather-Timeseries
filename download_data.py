import xarray as xr
import numpy as np
import os
from tqdm import trange
from time import time
import argparse


def log_time(func):
    """Decorator to log the time taken by a function."""

    def wrapper(*args, **kwargs):
        start_time = time()
        print(f"{func.__name__.replace('_', ' ').title()}: Processing", end="\r")
        result = func(*args, **kwargs)
        end_time = time()
        print(
            f"{func.__name__.replace('_', ' ').title()}: Complete - Time Taken: {end_time - start_time:.2f}"
        )
        return result

    return wrapper


@log_time
def load_dataset():
    return xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    )


def prepare_directory(base_dir, date_str, time_str):
    dir_path = os.path.join(base_dir, date_str, time_str)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def main(start_date, end_date, base_dir):
    ds = load_dataset()

    ds_filtered = ds.sel(time=slice(start_date, end_date))

    progress_bar = trange(len(ds_filtered["time"]), desc="Processing time steps")

    for i in progress_bar:
        current_time = ds_filtered["time"].isel(time=i).values
        date_str = np.datetime_as_string(current_time, unit="D")
        time_str = np.datetime_as_string(current_time, unit="m")[-5:]

        progress_bar.set_description(f"Saving Data: {date_str} {time_str}")

        dir_path = prepare_directory(base_dir, date_str, time_str)

        surface_vars = [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
        ]
        input_surface = np.stack(
            [
                ds_filtered[var].isel(time=i).values.astype(np.float32)
                for var in surface_vars
            ],
            axis=0,
        )

        np.save(os.path.join(dir_path, "input_surface.npy"), input_surface)

        upper_vars = [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ]
        input_upper = np.stack(
            [
                ds_filtered[var].isel(time=i).values.astype(np.float32)
                for var in upper_vars
            ],
            axis=0,
        )

        np.save(os.path.join(dir_path, "input_upper.npy"), input_upper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument(
        "--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--dir", type=str, default="data", help="Base directory for saving data"
    )

    args = parser.parse_args()

    main(args.start_date, args.end_date, args.dir)
