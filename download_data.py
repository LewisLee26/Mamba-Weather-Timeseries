import xarray as xr
import numpy as np
import argparse
from time import time
import pandas as pd


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
def process_and_save_data(
    output_path,
    start_date=None,
    end_date=None,
    years=None,
    months=None,
    days=None,
    times=None,
):
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
        chunks={"time": "auto"},
    )

    if start_date and end_date:
        ds_filtered = ds.sel(time=slice(start_date, end_date))
    elif years and months and days and times:
        datetime_strings = []
        for year in years:
            for month in months:
                for day in days:
                    for time_str in times:
                        datetime_strings.append(
                            f"{year:04d}-{month:02d}-{day:02d} {time_str}"
                        )
        try:
            datetime_index = pd.to_datetime(datetime_strings)
            ds_filtered = ds.sel(time=datetime_index)
        except ValueError as e:
            print(f"Error creating datetime index: {e}")
            return
    else:
        print(
            "Please provide either start_date and end_date, or years, months, days, and times."
        )
        return

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

    selected_data.to_zarr(output_path, mode="w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process weather data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start_date", type=str, help="Start date in YYYY-MM-DD format")
    group.add_argument(
        "--years", nargs="+", type=int, help="List of years (e.g., 2020 2021)"
    )
    parser.add_argument("--end_date", type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument(
        "--months", nargs="+", type=int, help="List of months (e.g., 1 2 3)"
    )
    parser.add_argument(
        "--days", nargs="+", type=int, help="List of days (e.g., 1 15 30)"
    )
    parser.add_argument(
        "--times", nargs="+", type=str, help="List of times (e.g., 00:00 06:00 12:00)"
    )
    parser.add_argument(
        "--output_path", type=str, default="output.zarr", help="Output Zarr file path"
    )

    args = parser.parse_args()

    if args.start_date:
        if not args.end_date:
            print("Error: end_date is required with start_date")
        else:
            process_and_save_data(
                args.output_path, start_date=args.start_date, end_date=args.end_date
            )
    else:
        if not (args.months and args.days and args.times):
            print("Error: months, days, and times are required with years")
        else:
            process_and_save_data(
                args.output_path,
                years=args.years,
                months=args.months,
                days=args.days,
                times=args.times,
            )

