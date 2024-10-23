import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from models.unet import UNet
from tqdm import tqdm


DATA_PATH = "data/processed"
BATCH_SIZE = 64
NUM_WORKERS = 1

NUM_EPOCHS = 200
PATIENCE = 10


def mwae_loss(y_pred, y_true, land_mask):
    # Convert land_mask to a PyTorch tensor if it's a NumPy array
    if isinstance(land_mask, np.ndarray):
        land_mask = torch.from_numpy(land_mask).to(y_pred.device)

    # Compute the absolute error
    absolute_error = torch.abs(y_pred - y_true)

    # Broadcast the land_mask to match the shape of y_true and y_pred
    land_mask_expanded = land_mask.unsqueeze(0).unsqueeze(1).expand_as(y_true)

    # Compute the weights
    weights = torch.ones_like(y_true)
    weights[land_mask_expanded == 0] = (
        0  # Set weight to 0 where there's land (masked area)
    )

    # Compute the weighted absolute error
    weighted_absolute_error = absolute_error * weights

    # Compute the MAE loss
    mae_loss = weighted_absolute_error.mean()

    return mae_loss


def cosine_distance_loss(y_true, y_pred, land_mask):
    # Normalize the vectors to compute cosine similarity
    y_true = F.normalize(y_true, p=2, dim=1)  # Normalize to unit vectors
    y_pred = F.normalize(y_pred, p=2, dim=1)

    # Compute cosine similarity
    cosine_similarity = torch.sum(y_true * y_pred, dim=1)

    # Broadcast the land_mask to match the shape of cosine_similarity
    # Here we expand to match batch_size, height, and width but without an extra channel dimension
    land_mask_expanded = land_mask.unsqueeze(0).expand(cosine_similarity.shape)

    # Apply the land mask: ignore land areas by setting their contribution to 0
    cosine_similarity[land_mask_expanded == 0] = 0

    # Compute the cosine distance loss (1 - cosine similarity)
    cosine_distance = 1 - cosine_similarity

    # Return the mean of the masked cosine distance
    return cosine_distance.mean()


def load_data():
    train_data = torch.load(f"{DATA_PATH}/train.pt")
    val_data = torch.load(f"{DATA_PATH}/validation.pt")
    test_data = torch.load(f"{DATA_PATH}/test.pt")
    normalization_data = torch.load(f"{DATA_PATH}/normalize.pt")

    return train_data, val_data, test_data, normalization_data


def create_land_mask(data):
    """
    Create a land mask based on the NaN values in the data.
    """
    # Construct a 13x17 matrix boolean mask for NaN values
    nan_mask = np.isnan(data[0, :, :, 0]).numpy()

    # Create a land mask by inverting the NaN mask
    land_mask = np.logical_not(nan_mask)

    return land_mask


def min_max_normalize(data, min_val=None, max_val=None, targets=[0, 1]):
    # Copy the data to avoid in-place modification
    normalized_data = np.copy(data)

    # Compute the min and max values for each target individually
    if min_val is None:
        min_val = data[:, :, :, targets].min(axis=(0, 1, 2), keepdims=True)
    if max_val is None:
        max_val = data[:, :, :, targets].max(axis=(0, 1, 2), keepdims=True)

    # Normalize the selected targets
    normalized_data[:, :, :, targets] = (
        normalized_data[:, :, :, targets] - min_val
    ) / (max_val - min_val)

    return normalized_data, min_val, max_val


def prepare_data(data, normalize, min_val=None, max_val=None, is_train=True):
    inputs = data["inputs"]
    outputs = data["outputs"]

    # Normalize the inputs
    input_means, input_stds = normalize["input_means"], normalize["input_stds"]
    inputs = (inputs - input_means) / input_stds
    outputs = np.nan_to_num(outputs, nan=0)

    if is_train:
        outputs, min_val, max_val = min_max_normalize(outputs, targets=[0, 1])
    else:
        outputs, _, _ = min_max_normalize(outputs, min_val, max_val, targets=[0, 1])

    # Convert the data to PyTorch tensors
    outputs = torch.from_numpy(outputs).float()

    dataset = TensorDataset(inputs, outputs)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    if is_train:
        return loader, min_val, max_val

    return loader


def get_loaders():
    train_data, val_data, test_data, normalization_data = load_data()
    land_mask = create_land_mask(train_data["outputs"])
    train_loader, min_val, max_val = prepare_data(train_data, normalization_data)
    val_loader = prepare_data(
        val_data, normalization_data, min_val, max_val, is_train=False
    )
    test_loader = prepare_data(
        test_data, normalization_data, min_val, max_val, is_train=False
    )

    print("Data processing completed successfully!")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    print(f"Land mask shape: {land_mask.shape}")

    return train_loader, val_loader, test_loader, land_mask, min_val, max_val


def train_model(
    model,
    train_loader,
    val_loader,
    land_mask,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    checkpoint_path="model_checkpoint.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    land_mask = torch.from_numpy(land_mask).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Training", total=num_epochs):
        model.train()
        total_swh_loss, total_mwp_loss, total_mwd_loss = 0, 0, 0
        total_train_loss = 0

        for inputs, outputs in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", total=len(train_loader)
        ):
            inputs, outputs = inputs.to(device), outputs.to(device)
            inputs = inputs.permute(0, 3, 1, 2)  # Permute to match model input format
            outputs = outputs.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            predictions = model(inputs)

            swh_pred, mwp_pred, mwd_sin_pred, mwd_cos_pred = torch.split(
                predictions, 1, dim=1
            )
            swh_true, mwp_true, mwd_sin_true, mwd_cos_true = torch.split(
                outputs, 1, dim=1
            )

            # Compute the loss for each output
            swh_loss = mwae_loss(swh_pred, swh_true, land_mask)
            mwp_loss = mwae_loss(mwp_pred, mwp_true, land_mask)
            mwd_loss = cosine_distance_loss(
                torch.cat((mwd_sin_pred, mwd_cos_pred), dim=1),
                torch.cat((mwd_sin_true, mwd_cos_true), dim=1),
                land_mask,
            )
            loss = swh_loss + mwp_loss + mwd_loss
            loss.backward()
            optimizer.step()

            total_swh_loss += swh_loss.item()
            total_mwp_loss += mwp_loss.item()
            total_mwd_loss += mwd_loss.item()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_swh_loss = total_swh_loss / len(train_loader)
        avg_mwp_loss = total_mwp_loss / len(train_loader)
        avg_mwd_loss = total_mwd_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_swh_loss, val_mwp_loss, val_mwd_loss = 0, 0, 0
        val_loss = 0
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = outputs.permute(0, 3, 1, 2)
                predictions = model(inputs)

                swh_pred, mwp_pred, mwd_sin_pred, mwd_cos_pred = torch.split(
                    predictions, 1, dim=1
                )
                swh_true, mwp_true, mwd_sin_true, mwd_cos_true = torch.split(
                    outputs, 1, dim=1
                )

                swh_loss = mwae_loss(swh_pred, swh_true, land_mask)
                mwp_loss = mwae_loss(mwp_pred, mwp_true, land_mask)
                mwd_loss = cosine_distance_loss(
                    torch.cat((mwd_sin_pred, mwd_cos_pred), dim=1),
                    torch.cat((mwd_sin_true, mwd_cos_true), dim=1),
                    land_mask,
                )
                val_loss += swh_loss + mwp_loss + mwd_loss

                val_swh_loss += swh_loss.item()
                val_mwp_loss += mwp_loss.item()
                val_mwd_loss += mwd_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_swh_loss = val_swh_loss / len(val_loader)
            avg_val_mwp_loss = val_mwp_loss / len(val_loader)
            avg_val_mwd_loss = val_mwd_loss / len(val_loader)

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return model


def test_model(model, test_loader, land_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    land_mask = torch.from_numpy(land_mask).to(device)
    total_swh_loss, total_mwp_loss, total_mwd_loss = 0, 0, 0
    total_loss = 0

    model.to(device)

    model.eval()

    with torch.no_grad():
        for inputs, outputs in test_loader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = outputs.permute(0, 3, 1, 2)
            predictions = model(inputs)

            swh_pred, mwp_pred, mwd_sin_pred, mwd_cos_pred = torch.split(
                predictions, 1, dim=1
            )
            swh_true, mwp_true, mwd_sin_true, mwd_cos_true = torch.split(
                outputs, 1, dim=1
            )

            swh_loss = mwae_loss(swh_pred, swh_true, land_mask)
            mwp_loss = mwae_loss(mwp_pred, mwp_true, land_mask)
            mwd_loss = cosine_distance_loss(
                torch.cat((mwd_sin_pred, mwd_cos_pred), dim=1),
                torch.cat((mwd_sin_true, mwd_cos_true), dim=1),
                land_mask,
            )
            loss = swh_loss + mwp_loss + mwd_loss

            total_swh_loss += swh_loss.item()
            total_mwp_loss += mwp_loss.item()
            total_mwd_loss += mwd_loss.item()
            total_loss += loss.item()

        avg_swh_loss = total_swh_loss / len(test_loader)
        avg_mwp_loss = total_mwp_loss / len(test_loader)
        avg_mwd_loss = total_mwd_loss / len(test_loader)
        avg_loss = total_loss / len(test_loader)

    print(
        f"Test loss: {avg_loss:.4f}, SWH loss: {avg_swh_loss:.4f}, MWP loss: {avg_mwp_loss:.4f}, MWD loss: {avg_mwd_loss:.4f}"
    )
    return avg_loss, avg_swh_loss, avg_mwp_loss, avg_mwd_loss


def test_model_with_denormalization(
    model, test_loader, land_mask, min_swh, max_swh, min_mwp, max_mwp
):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert min and max values to PyTorch tensors
    min_swh = torch.tensor(min_swh).float().to(device)
    max_swh = torch.tensor(max_swh).float().to(device)
    min_mwp = torch.tensor(min_mwp).float().to(device)
    max_mwp = torch.tensor(max_mwp).float().to(device)

    # Convert land_mask to a PyTorch tensor if it's a NumPy array
    land_mask = torch.from_numpy(land_mask).to(device)
    H, W = land_mask.shape

    # Initialize variables to store the total error and the number of ocean points
    total_swh_error, total_mwp_error, total_mwd_error = 0, 0, 0
    total_ocean_points = 0

    # Initialize accumulators for per-grid-point errors and counts
    total_swh_error_map = torch.zeros([H, W], device=device)
    total_mwp_error_map = torch.zeros([H, W], device=device)
    total_mwd_error_map = torch.zeros([H, W], device=device)
    count_map = torch.zeros([H, W], device=device)

    with torch.no_grad():
        for inputs, outputs in test_loader:

            # Move the data to the appropriate device
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            # Permute the input and output tensors to match the model input format
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = outputs.permute(0, 3, 1, 2)

            # Get the model predictions
            predictions = model(inputs)

            # Split the predictions and outputs into individual variables
            swh_pred, mwp_pred, mwd_sin_pred, mwd_cos_pred = torch.split(
                predictions, 1, dim=1
            )
            swh_true, mwp_true, mwd_sin_true, mwd_cos_true = torch.split(
                outputs, 1, dim=1
            )

            # Denormalize the predicted and true values
            swh_pred_denorm = swh_pred * (max_swh - min_swh) + min_swh
            mwp_pred_denorm = mwp_pred * (max_mwp - min_mwp) + min_mwp
            swh_true_denorm = swh_true * (max_swh - min_swh) + min_swh
            mwp_true_denorm = mwp_true * (max_mwp - min_mwp) + min_mwp

            # Compute the absolute error for each output
            swh_abs_error = torch.abs(swh_pred_denorm - swh_true_denorm).squeeze(
                1
            )  # Shape [batch_size, H, W]
            mwp_abs_error = torch.abs(mwp_pred_denorm - mwp_true_denorm).squeeze(1)

            # Compute the true and predicted angles from the sine and cosine values
            mwd_pred_angle = torch.atan2(mwd_sin_pred, mwd_cos_pred)
            mwd_true_angle = torch.atan2(mwd_sin_true, mwd_cos_true)

            # Compute the absolute difference between the predicted and true angles
            mwd_diff = torch.abs(
                torch.atan2(
                    torch.sin(mwd_pred_angle - mwd_true_angle),
                    torch.cos(mwd_pred_angle - mwd_true_angle),
                )
            ).squeeze(1)
            mwd_diff_deg = mwd_diff * (180.0 / np.pi)  # Convert to degrees

            # Apply the land mask to the absolute error tensors
            land_mask_expanded = land_mask.unsqueeze(0)  # Shape [1, H, W]
            swh_error_batch = swh_abs_error * land_mask_expanded
            mwp_error_batch = mwp_abs_error * land_mask_expanded
            mwd_error_batch = mwd_diff_deg * land_mask_expanded

            # Compute the number of ocean points in the batch
            ocean_points = land_mask_expanded.expand_as(swh_abs_error).sum()

            # Update the total number of ocean points
            total_ocean_points += ocean_points.item()

            # Sum the errors over the batch dimension
            total_swh_error += swh_error_batch.sum().item()
            total_mwp_error += mwp_error_batch.sum().item()
            total_mwd_error += mwd_error_batch.sum().item()

            # Accumulate per-grid-point errors
            total_swh_error_map += swh_error_batch.sum(dim=0)  # Shape [H, W]
            total_mwp_error_map += mwp_error_batch.sum(dim=0)
            total_mwd_error_map += mwd_error_batch.sum(dim=0)

            # Accumulate counts
            counts_batch = land_mask_expanded * torch.ones_like(swh_abs_error)
            count_map += counts_batch.sum(dim=0)

        # Compute the average error values
        avg_swh_error = total_swh_error / total_ocean_points
        avg_mwp_error = total_mwp_error / total_ocean_points
        avg_mwd_error = total_mwd_error / total_ocean_points

    # Compute the mean error per grid point
    mean_swh_error_map = total_swh_error_map / count_map
    mean_mwp_error_map = total_mwp_error_map / count_map
    mean_mwd_error_map = total_mwd_error_map / count_map

    # Handle division by zero (land points)
    mean_swh_error_map[count_map == 0] = np.nan
    mean_mwp_error_map[count_map == 0] = np.nan
    mean_mwd_error_map[count_map == 0] = np.nan

    print(f"Test Errors (in original units):")
    print(f"SWH MAE: {avg_swh_error:.4f} meters")
    print(f"MWP MAE: {avg_mwp_error:.4f} seconds")
    print(f"MWD MAE: {avg_mwd_error:.4f} degrees")

    # Plot and save the error maps
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(mean_swh_error_map.cpu().numpy(), cmap="jet", origin="lower")
    plt.colorbar()
    plt.title("Mean SWH Error (m)")
    plt.savefig("mean_swh_error_map.png")

    plt.figure()
    plt.imshow(mean_mwp_error_map.cpu().numpy(), cmap="jet", origin="lower")
    plt.colorbar()
    plt.title("Mean MWP Error (s)")
    plt.savefig("mean_mwp_error_map.png")

    plt.figure()
    plt.imshow(mean_mwd_error_map.cpu().numpy(), cmap="jet", origin="lower")
    plt.colorbar()
    plt.title("Mean MWD Error (degrees)")
    plt.savefig("mean_mwd_error_map.png")

    return avg_swh_error, avg_mwp_error, avg_mwd_error


if __name__ == "__main__":
    train_loader, val_loader, test_loader, land_mask, min_val, max_val = get_loaders()

    print(min_val, max_val)

    # Step 1: Instantiate the UNet model
    model = UNet()

    # Load the model checkpoint if it exists
    checkpoint_path = "model_checkpoint.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded model checkpoint")
    except FileNotFoundError:
        print("Model checkpoint not found. Training a new model...")
        model = train_model(
            model,
            train_loader,
            val_loader,
            land_mask,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            checkpoint_path=checkpoint_path,
        )

    # Step 3: Test the model on the test dataset
    test_model(model, test_loader, land_mask)

    # [[[[0. 0.]]]] [[[[ 9.549961 11.274692]]]]
    min_swh, max_swh = (0, 9.549961)
    min_mwp, max_mwp = (0, 11.274692)

    test_model_with_denormalization(
        model, test_loader, land_mask, min_swh, max_swh, min_mwp, max_mwp
    )
