import xarray as xr
import numpy as np
import argparse
from time import time
import gcsfs


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
    # Define the URL of the big Zarr dataset
    zarr_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

    # Open the Zarr dataset with auto-chunking on time, letting fsspec handle the GCS URL
    ds = xr.open_zarr(zarr_url, chunks={"time": "auto"})

    # Select only the requested time slice
    ds_filtered = ds.sel(time=slice(start_date, end_date))

    # Define which variables we care about
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

    # Subset and cast to float32 for intermediate
    selected = ds_filtered[surface_vars + upper_vars].astype(np.float32)

    # Write the processed data to the specified GCS path
    selected.to_netcdf(args.output_path, mode=args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process weather data.")
    parser.add_argument(
        "--start_date", type=str, required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end_date", type=str, required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/input.nc",
        help="Output GCS Zarr file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="w",
    )

    args = parser.parse_args()
    process_and_save_data(args.start_date, args.end_date, args.output_path)
