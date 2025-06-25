import torch
from shared import FEATURES, MODEL_PATH, MODEL_STRUCT, SCALER_PATH, get_data
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


def predict_top_players(model, data, num_players=3):
    model.eval()

    # Load pre-fitted scaler
    scaler = joblib.load(SCALER_PATH)
    x_test = scaler.transform(data[FEATURES])  # Only transform — do not fit again
    x_test = torch.tensor(x_test, dtype=torch.float32)

    # Move input to same device as model
    device = next(model.parameters()).device
    x_test = x_test.to(device)

    # Predict probabilities
    with torch.no_grad():
        probabilities = model(x_test).squeeze().cpu().numpy()  # back to CPU for numpy
    data["probability"] = probabilities

    # Select top player per (date, tims) group
    filtered_data = data[data["tims"] != 0].dropna(subset=["tims"])
    top_players_per_day = (
        filtered_data
        .groupby(["date", "tims"], as_index=False)
        .apply(lambda group: group.nlargest(1, "probability"))
        .reset_index(drop=True)
    )

    return top_players_per_day.reset_index(drop=True)


def evaluate_predictions(top_players):
    correct_predictions = (top_players["scored"] == 1).sum()
    total_predictions = len(top_players)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions


def test_model(data, labels, model_path=MODEL_PATH):
    data = data.copy()
    data["scored"] = labels

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and move to device
    model = MODEL_STRUCT.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Run predictions
    top_players = predict_top_players(model, data)

    print("\nTop Players Per Day:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(top_players[["date", "name", "probability", "scored"]])

    # Evaluate
    accuracy, correct, total = evaluate_predictions(top_players)
    print(f"Top 3 Prediction Accuracy: {accuracy:.2f}")
    print(f"Correct Predictions: {correct}/{total}")


if __name__ == "__main__":
    data, labels = get_data()
    test_model(data, labels)
