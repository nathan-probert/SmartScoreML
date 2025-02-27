import torch
from constants import FEATURES, MODEL_PATH, MODEL_STRUCT
from sklearn.preprocessing import StandardScaler


def get_model():
    model = MODEL_STRUCT
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    return model


def apply_model(data):
    model = get_model()

    scaler = StandardScaler()
    x_test = scaler.fit_transform(data[FEATURES])
    x_test = torch.tensor(x_test, dtype=torch.float32)

    with torch.no_grad():
        probabilities = model(x_test).squeeze().numpy()
    data["probability"] = probabilities

    return data
