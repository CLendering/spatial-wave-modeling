import numpy as np
import xarray as xr
import torch
import zipfile
import os
import tqdm
from torch.utils.data import Dataset, DataLoader

# Constants
DATA_DIR = "data/era5/"
OUTPUT_DIR = "data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_YEARS = list(range(1979, 2010))
VAL_YEARS = list(range(2010, 2015))
TEST_YEARS = list(range(2015, 2024))

ALL_YEARS = range(1979, 2024)

BATCH_SIZE = 16
NUM_WORKERS = 1

SAVE_TENSORS = True


def unzip_and_organize_data():
    """
    Unzips the downloaded ERA5 data and organizes it into separate files for winds and waves.
    """
    for year in tqdm.tqdm(ALL_YEARS, desc="Processing data", unit="year"):
        wind_file = f"{DATA_DIR}unzipped/winds_{year}.nc"
        wave_file = f"{DATA_DIR}unzipped/waves_{year}.nc"

        if os.path.exists(wind_file) and os.path.exists(wave_file):
            print(f"Data for year {year} is already unzipped and organized.")
            continue

        os.makedirs(f"{DATA_DIR}unzipped/", exist_ok=True)
        try:
            with zipfile.ZipFile(
                f"{DATA_DIR}downloaded/era5_adriatic_{year}.nc", "r"
            ) as zip_ref:
                zip_ref.extractall(f"{DATA_DIR}unzipped/")
            os.rename(f"{DATA_DIR}unzipped/data_stream-oper.nc", wind_file)
            os.rename(f"{DATA_DIR}unzipped/data_stream-wave.nc", wave_file)
        except (zipfile.BadZipFile, FileNotFoundError, OSError) as e:
            print(f"Error processing data for the year {year}: {e}")


def load_wind_data(year):
    """
    Loads the wind data for a given year from 'winds_{year}.nc' file.
    """
    file_path = os.path.join(DATA_DIR, "unzipped", f"winds_{year}.nc")
    try:
        data = xr.open_dataset(file_path)
        u10 = data["u10"]
        v10 = data["v10"]
        return u10, v10
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError as e:
        print(f"Key error: {e} in file {file_path}")
    except Exception as e:
        print(f"An error occurred while loading wind data for the year {year}: {e}")
    return None, None


def load_wave_data(year):
    """
    Loads the wave data for a given year from 'waves_{year}.nc' file.
    """
    file_path = os.path.join(DATA_DIR, "unzipped", f"waves_{year}.nc")
    try:
        data = xr.open_dataset(file_path)
        swh = data["swh"]
        mwd = data["mwd"]
        mwp = data["mwp"]

        # Convert wave direction to sine and cosine components
        mwd_rad = np.deg2rad(mwd)
        mwd_sin = np.sin(mwd_rad)
        mwd_cos = np.cos(mwd_rad)

        return swh, mwp, mwd_sin, mwd_cos
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError as e:
        print(f"Key error: {e} in file {file_path}")
    except Exception as e:
        print(f"An error occurred while loading wave data for the year {year}: {e}")
    return None, None, None, None


def process_year(year):
    """
    Process the wind and wave data for a given year.
    """
    u10, v10 = load_wind_data(year)
    swh, mwp, mwd_sin, mwd_cos = load_wave_data(year)
    if (
        u10 is None
        or v10 is None
        or swh is None
        or mwp is None
        or mwd_sin is None
        or mwd_cos is None
    ):
        raise ValueError(f"Data not found for year {year}")

    inputs = np.stack([u10.values, v10.values], axis=-1)
    outputs = np.stack(
        [swh.values, mwp.values, mwd_sin.values, mwd_cos.values], axis=-1
    )

    # Print the largest mwp value in the dataset but habndle nan values
    mwp = mwp.where(~np.isnan(mwp), -100)
    print(f"Max mean wave period: {np.max(mwp.values)}")

    return inputs, outputs


class WindWaveDataset(Dataset):
    """
    Custom PyTorch dataset for wind and wave data.
    """

    def __init__(self, inputs, outputs, transform=None):
        self.inputs = inputs
        self.outputs = outputs
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]

        if self.transform:
            input_data, output_data = self.transform(input_data, output_data)

        return input_data, output_data


class NormalizeTransform:
    """
    Normalizes the input and output data.
    """

    def __init__(self, input_means, input_stds, output_means, output_stds):
        self.input_means = input_means
        self.input_stds = input_stds
        self.output_means = output_means
        self.output_stds = output_stds

    def __call__(self, input_data, output_data):
        input_data = (input_data - self.input_means) / self.input_stds
        output_data = (output_data - self.output_means) / self.output_stds
        return input_data, output_data


def main():
    unzip_and_organize_data()

    train_input_list, train_output_list = [], []
    val_input_list, val_output_list = [], []
    test_input_list, test_output_list = [], []

    for year in tqdm.tqdm(TRAIN_YEARS, desc="Processing training data", unit="year"):
        try:
            inputs, outputs = process_year(year)
            train_input_list.append(inputs)
            train_output_list.append(outputs)
        except ValueError as e:
            print(e)

    train_inputs = np.concatenate(train_input_list, axis=0)
    train_outputs = np.concatenate(train_output_list, axis=0)

    for year in tqdm.tqdm(VAL_YEARS, desc="Processing validation data", unit="year"):
        try:
            inputs, outputs = process_year(year)
            val_input_list.append(inputs)
            val_output_list.append(outputs)
        except ValueError as e:
            print(e)

    val_inputs = np.concatenate(val_input_list, axis=0)
    val_outputs = np.concatenate(val_output_list, axis=0)

    for year in tqdm.tqdm(TEST_YEARS, desc="Processing test data", unit="year"):
        try:
            inputs, outputs = process_year(year)
            test_input_list.append(inputs)
            test_output_list.append(outputs)
        except ValueError as e:
            print(e)

    test_inputs = np.concatenate(test_input_list, axis=0)
    test_outputs = np.concatenate(test_output_list, axis=0)

    # Normalize the input and output data
    input_flat = train_inputs.reshape(-1, 2)
    input_means = input_flat.mean(axis=0)
    input_stds = input_flat.std(axis=0)

    output_flat = train_outputs.reshape(-1, 4)
    output_means = output_flat.mean(axis=0)
    output_stds = output_flat.std(axis=0)

    normalize_transform = NormalizeTransform(
        input_means, input_stds, output_means, output_stds
    )

    train_dataset = WindWaveDataset(
        train_inputs, train_outputs, transform=normalize_transform
    )
    val_dataset = WindWaveDataset(
        val_inputs, val_outputs, transform=normalize_transform
    )
    test_dataset = WindWaveDataset(
        test_inputs, test_outputs, transform=normalize_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print("Data processing completed successfully!")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    if SAVE_TENSORS:
        train_data = {
            "inputs": torch.from_numpy(train_inputs).float(),
            "outputs": torch.from_numpy(train_outputs).float(),
        }
        torch.save(train_data, os.path.join(OUTPUT_DIR, "train.pt"))

        val_data = {
            "inputs": torch.from_numpy(val_inputs).float(),
            "outputs": torch.from_numpy(val_outputs).float(),
        }
        torch.save(val_data, os.path.join(OUTPUT_DIR, "validation.pt"))

        test_data = {
            "inputs": torch.from_numpy(test_inputs).float(),
            "outputs": torch.from_numpy(test_outputs).float(),
        }
        torch.save(test_data, os.path.join(OUTPUT_DIR, "test.pt"))

        normalization_data = {
            "input_means": torch.from_numpy(input_means).float(),
            "input_stds": torch.from_numpy(input_stds).float(),
            "output_means": torch.from_numpy(output_means).float(),
            "output_stds": torch.from_numpy(output_stds).float(),
        }
        torch.save(normalization_data, os.path.join(OUTPUT_DIR, "normalize.pt"))

        print("Data saved as tensors successfully!")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
