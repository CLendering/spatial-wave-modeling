import cdsapi
import os
from tqdm import tqdm
import time

OUTPUT_DIR = "data/era5/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 1980
END_YEAR = 2023

c = cdsapi.Client()

# Variables to download
variables = [
    "10m_u_component_of_wind",  # U-component of wind
    "10m_v_component_of_wind",  # V-component of wind
    "significant_height_of_combined_wind_waves_and_swell",  # Wave height
    "mean_wave_direction",  # Wave direction
    "mean_wave_period",  # Wave period
]

years = range(START_YEAR, END_YEAR + 1)

for year in tqdm(years, desc="Downloading ERA5 data", unit="year"):
    print(f"Downloading data for the year {year}...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": str(year),
            "month": [
                "".join(["0", str(month)])[-2:] for month in range(1, 13)
            ],  # 12 months
            "day": [
                "".join(["0", str(day)])[-2:] for day in range(1, 32)
            ],  # 31 days for all months
            "time": [  # 3-hourly data
                "00:00",
                "03:00",
                "06:00",
                "09:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00",
            ],
            "area": [46, 12, 40, 20],
            "format": "netcdf",
        },
        f"{OUTPUT_DIR}era5_adriatic_{year}.nc",
    )
    time.sleep(5)
print("Data downloaded successfully!")
