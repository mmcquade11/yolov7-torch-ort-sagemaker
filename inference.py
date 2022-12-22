import os.path
from time import perf_counter

import os
import torch
import io
import logging

from sagemaker_inference import content_types, encoder, errors, utils
from PIL import Image
from torchvision import transforms

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)


from models.experimental import attempt_load

def model_fn(model_dir):
    from torch_ort import ORTInferenceModule
    logging.info("Loading model")
    path = os.path.join(model_dir, 'model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(path, map_location=device)
    return (model,ORTInferenceModule(model))



def input_fn(input_data, content_type):
    """
    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    """
    if content_type == "application/x-image":
        decoded = Image.open(io.BytesIO(input_data))
    else:
        raise ValueError(f"Type [{content_type}] not supported.")

    preprocess = transforms.Compose([transforms.ToTensor()])
    normalized = preprocess(decoded)
    return normalized


def predict_fn(data, model):
    """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
    Runs prediction on GPU if cuda is available.
    Args:
        data: input data (torch.Tensor) for prediction deserialized by input_fn
        model: PyTorch model loaded in memory by model_fn
    Returns: a prediction
    """
    with torch.no_grad():
        if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
            device = torch.device("cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                output = model([input_data])
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            output = model([input_data])

    return output


def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.
    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized
    Returns: output data serialized
    """
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy().tolist()

    for content_type in utils.parse_accept(accept):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(prediction, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    raise errors.UnsupportedFormatError(accept)

