import torch
from shared import FEATURES, MODEL_PATH, MODEL_STRUCT, ONNX_MODEL_PATH, get_data
from sklearn.preprocessing import StandardScaler
from test_model import test_model
from torch import nn, optim

# We cannot use traditional testing for this binary classification model
# because the model will (correctly) always predict the player will not score
# since the only true testing I care about is our accuracy in predicting goal scorers
# probability, i.e choosing the best possible available pick, I will test the model by
# picking the top 3 picks for each day and seeing how many of those players actually scored.
# This means that I will be testing with training data, so overfitting is a concern. However
# this is only temporary, as I will be recording Tim's picks and using that as the test data

# Once I start recording tims picks, I can use that to test the model. This will be available
# soon as I can reserve those days for testing, and use the rest for training


def create_model(data, labels):
    # Prepare the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data[FEATURES])

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(labels.values, dtype=torch.float32)

    # Define model
    model = MODEL_STRUCT

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 500

    patience = 20
    best_loss = float("inf")
    no_improvement = 0

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Early stopping logic
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)  # Save the best model
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), MODEL_PATH)
    torch.onnx.export(model, x_train, ONNX_MODEL_PATH, opset_version=11)


def main():
    data, labels = get_data()

    create_model(data, labels)
    test_model(data)


if __name__ == "__main__":
    main()
