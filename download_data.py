import xarray as xr
import numpy as np
import argparse
from time import time


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
def process_and_save_data(start_date, end_date, output_path):
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        chunks={"time": "auto"},  # Let xarray decide the chunks
    )

    ds_filtered = ds.sel(time=slice(start_date, end_date))

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

    selected_data = ds_filtered[surface_vars + upper_vars].astype(np.float32)

    selected_data.to_zarr(output_path, mode="w")  # Save as Zarr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument(
        "--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--output_path", type=str, default="output.zarr", help="Output Zarr file path"
    )

    args = parser.parse_args()

    process_and_save_data(args.start_date, args.end_date, args.output_path)
