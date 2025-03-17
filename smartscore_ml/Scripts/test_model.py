import torch
from shared import FEATURES, MODEL_PATH, MODEL_STRUCT, get_data
from sklearn.preprocessing import StandardScaler


def predict_top_players(model, data, num_players=3):
    scaler = StandardScaler()
    x_test = scaler.fit_transform(data[FEATURES])
    x_test = torch.tensor(x_test, dtype=torch.float32)

    with torch.no_grad():
        probabilities = model(x_test).squeeze().numpy()
    data["probability"] = probabilities

    # top_players_per_day = data.groupby("date")[["date", "name", "probability", "scored", "tims"]].apply(
    #     lambda group: group.nlargest(num_players, "probability")
    # )
    # return top_players_per_day.reset_index(drop=True)
    top_players_per_day = (
        data[data["tims"] != 0]
        .dropna(subset=["tims"])
        .groupby("date")
        .apply(
            lambda group: group.groupby("tims")
            .apply(lambda sub_group: sub_group.nlargest(1, "probability"))
            .reset_index(drop=True)
        )
    )
    return top_players_per_day.reset_index(drop=True)


def evaluate_predictions(top_players):
    correct_predictions = 0
    total_predictions = len(top_players)

    for _, row in top_players.iterrows():
        if row["scored"] == 1:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions


def test_model(data, model_path=MODEL_PATH):
    model = MODEL_STRUCT
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    top_players = predict_top_players(model, data)

    # Output the top players and their probabilities for analysis
    print("\nTop Players Per Day:")
    print(top_players)

    # Evaluate predictions
    accuracy, correct, total = evaluate_predictions(top_players)
    print(f"Top 3 Prediction Accuracy: {accuracy:.2f}")
    print(f"Correct Predictions: {correct}/{total}")


if __name__ == "__main__":
    data, labels = get_data()

    test_model(data)
