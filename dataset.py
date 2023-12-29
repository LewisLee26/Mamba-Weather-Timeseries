import requests
from tqdm import tqdm
import datetime
import pygrib
import pandas as pd
import os

current_date = datetime.datetime(2015, 1, 15)
end_date = datetime.datetime(2015, 2, 15)
# current_year = datetime.datetime.today().year
# current_month = datetime.datetime.today().month
# current_day = datetime.datetime.today().day
# end_date = datetime.datetime(current_year, current_month, current_day)

coord_index = 0

columns = [
    "coord_index",
    "lat",
    "lon",
    "Temperature",
    "Surface pressure",
    "V component of wind",
    "U component of wind",
    "Specific humidity",
    "Convective precipitation",
    "Total precipitation",
    "Water equivalent of accumulated snow depth",
    # "Soil Temperature",
    # "Volumetric soil moisture content",
]

df = pd.DataFrame(columns=columns)

pbar = tqdm(total = (1 * 30 * 4 * (10 * 20)))

while True:
    if end_date == current_date:
        break
    url = current_date.strftime(r"%Y/%Y%m%d/gfs.0p25.%Y%m%d%H.f003.grib2")

    idx = url.rfind("/")
    if (idx > 0):
        ofile = url[idx+1:]
    else:
        ofile = url

    try:
        response = requests.get("https://data.rda.ucar.edu/ds084.1/" + url)
    except Exception as e:
        print(e)
        break

    with open(ofile, "wb") as f:
        f.write(response.content)
    f.close()

    grbs = pygrib.open(ofile)

    max_x = 10
    max_y = 20
    for x in range(max_x): # 721
        for y in range(max_y):
            lat = round((y/max_y)*720)
            lon = round((x/max_x)*1439)
            
            df.loc[-1] = [
                coord_index,
                lat,
                lon,
                grbs[219].values[lat][lon], # Temperature
                grbs[217].values[lat][lon], # Surface pressure
                grbs[2].values[lat][lon], # V component of wind
                grbs[1].values[lat][lon], # U component of wind
                grbs[232].values[lat][lon], # Specific humidity
                grbs[244].values[lat][lon], # Convective precipitation
                grbs[243].values[lat][lon], # Total precipitation
                grbs[228].values[lat][lon], # Water equivalent of accumlated snow depth
                # grbs[226].values[lat][lon], # Soil temperature
                # grbs[227].values[lat][lon], # Volumeric soil moisture content
            ]  

            # shifting index
            df.index = df.index + 1 
            # sorting by index
            df = df.sort_index()  
            
            pbar.update(1)

    grbs.close()

    checkpoint = 37
    if coord_index % checkpoint == 0:
        df.to_parquet(f"gfs_dataframe_{coord_index//checkpoint}.parquet")

    coord_index +=1
    os.remove(ofile)
    current_date += datetime.timedelta(hours=6)

df.to_parquet(f"gfs_dataframe.parquet")

