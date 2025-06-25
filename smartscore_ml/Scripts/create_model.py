from shared import FEATURES, MODEL_PATH, MODEL_STRUCT, ONNX_MODEL_PATH, SCALER_PATH, get_data
from test_model import test_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib


def create_model(data, labels):
    # Hyperparameters
    epochs = 250
    batch_size = 64
    learning_rate = 0.001

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalize features
    scaler = StandardScaler()
    x_all = scaler.fit_transform(data[FEATURES])
    joblib.dump(scaler, SCALER_PATH)

    # Convert to tensors
    x_all_tensor = torch.tensor(x_all, dtype=torch.float32)
    y_all_tensor = torch.tensor(labels.values, dtype=torch.float32)

    # Validation split (e.g., 80% train, 20% val)
    val_split = 0.2
    num_samples = len(x_all_tensor)
    split_idx = int((1 - val_split) * num_samples)

    x_train_tensor = x_all_tensor[:split_idx].to(device)
    y_train_tensor = y_all_tensor[:split_idx].to(device)
    x_val_tensor = x_all_tensor[split_idx:].to(device)
    y_val_tensor = y_all_tensor[split_idx:].to(device)

    # Training DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = MODEL_STRUCT.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation loss (evaluated on entire val set)
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()
        model.train()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

    # Save trained model
    torch.save(model.state_dict(), MODEL_PATH)


def split_data_by_date(data, labels, train_fraction=0.8):
    unique_dates = sorted(data["date"].unique())
    split_index = int(train_fraction * len(unique_dates))

    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_data = data[data["date"].isin(train_dates)].reset_index(drop=True)
    test_data = data[data["date"].isin(test_dates)].reset_index(drop=True)

    train_labels = labels.iloc[train_data.index]
    test_labels = labels.iloc[test_data.index]

    return train_data, train_labels, test_data, test_labels


def main():
    data, labels = get_data()

    data = data.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    train_data, train_labels, test_data, test_labels = split_data_by_date(data, labels)

    create_model(train_data, train_labels)
    test_model(test_data, test_labels)


if __name__ == "__main__":
    main()
