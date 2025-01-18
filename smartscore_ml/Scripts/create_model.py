import base64
import csv
import gzip
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import boto3
import json
from torch.utils.data import DataLoader, TensorDataset


session = boto3.Session()
sts_client = boto3.client("sts")
lambda_client = boto3.client("lambda")


def invoke_lambda(function_name, payload, wait=True):
    session = boto3.session.Session()
    region = session.region_name
    account_id = sts_client.get_caller_identity()["Account"]
    invocation_type = "RequestResponse" if wait else "Event"

    function_arn = f"arn:aws:lambda:{region}:{account_id}:function:{function_name}"
    response = lambda_client.invoke(
        FunctionName=function_arn, InvocationType=invocation_type, Payload=json.dumps(payload)
    )
    response_payload = json.loads(response["Payload"].read())
    return response_payload


def unpack_response(body):
    compressed_data = base64.b64decode(body)
    decompressed_data = gzip.decompress(compressed_data).decode('utf-8')
    original_data = json.loads(decompressed_data)

    return original_data


def create_model():
    path = "smartscore_ml\\lib"

    # Load the data
    print("Do you want to download the data from the database? (y/n)")
    choice = input().split()[0].lower()

    if choice == "y":
        response = invoke_lambda("Api-prod", {"method": "GET_ALL"})
        data = unpack_response(response.get("entries"))

        # Get the fields from the last entry (which should have all fields)
        all_fields = data[-1].keys()
        for entry in data:
            # Add missing keys with None
            for field in all_fields:
                entry.setdefault(field, None)

        os.makedirs(path, exist_ok=True)
        with open(f"{path}\\data.csv", "w+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(data)

    data = pd.read_csv(f"{path}\\data.csv", encoding="utf-8")

    data = data[data['scored'] != ' ']
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Preprocess the labels to binary (0 or 1)
    labels = data['scored'].apply(lambda x: 1 if x == 'scored' else 0)  # Adjust as needed

    features = data[['gpg', 'hgpg', 'five_gpg', 'tgpg', 'otga']]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=None, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Define model
    input_size = x_train.shape[1]
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()  # Apply sigmoid activation for binary output
    )

    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs.squeeze(), y_train)  # Make sure outputs and targets match shapes
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
        accuracy = (predicted == y_test).float().mean()
        print(f'Test Accuracy: {accuracy:.4f}')

    torch.save(model.state_dict(), f"{path}\\model.pth")


if __name__ == "__main__":
    create_model()

