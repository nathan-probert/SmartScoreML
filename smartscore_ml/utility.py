import onnxruntime as ort
from constants import FEATURES, ONNX_MODEL_PATH
from sklearn.preprocessing import StandardScaler


def get_model_onnx():
    """
    Load the ONNX model for inference using ONNX Runtime.
    """    
    # Load the ONNX model using ONNX Runtime
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    return ort_session


def apply_model(data):
    """
    Apply the ONNX model to the data using ONNX Runtime.
    """
    # Prepare the data
    scaler = StandardScaler()
    x_test = scaler.fit_transform(data[FEATURES])

    # Load the ONNX model
    ort_session = get_model_onnx()

    # Perform inference with ONNX Runtime
    inputs = {ort_session.get_inputs()[0].name: x_test.astype('float32')}
    probabilities = ort_session.run(None, inputs)[0].squeeze()

    # Add the predictions to the dataframe
    data["probability"] = probabilities

    return data
