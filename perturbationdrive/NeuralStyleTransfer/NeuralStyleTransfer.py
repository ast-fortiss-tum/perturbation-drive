import cv2
import os
from ..utils import download_file


class NeuralStyleTransfer:
    """
    Transforms the style of the image to one of 9 different
    artistic styles.

    On init the user can choose all styles to load and via
    `transferStyle` to user can transfer the image style to
    on of the preloaded styles

    All eccv16 models are too slow and should not be used
    """

    def __init__(
        self,
        model_names=[
            "perturbationdrive/NeuralStyleTransfer/models/instance_norm/candy.t7",
            "perturbationdrive/NeuralStyleTransfer/models/instance_norm/feathers.t7",
            "perturbationdrive/NeuralStyleTransfer/models/instance_norm/la_muse.t7",
            "perturbationdrive/NeuralStyleTransfer/models/instance_norm/mosaic.t7",
            "perturbationdrive/NeuralStyleTransfer/models/instance_norm/the_scream.t7",
            "perturbationdrive/NeuralStyleTransfer/models/instance_norm/udnie.t7",
        ],
    ):
        self.models = {}
        for path in model_names:
            model_name = os.path.splitext(os.path.basename(path))[0]
            if not os.path.exists(path):
                # download and save file
                parts = path.split("/")
                model_url = (
                    "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/"
                    + "/".join(parts[-2:])
                )
                print(f"Fetching Neural Style Transfer {model_name} from {model_url}")
                download_file(model_url, path)
            else:
                print(f"Loading Neural Style Transfer {model_name} from {path}")
            self.models[model_name] = cv2.dnn.readNetFromTorch(path)

    def transferStyle(self, image, model_name):
        model = self.models[model_name]
        if model == None:
            return image
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False
        )
        model.setInput(blob)
        output = model.forward()
        output = output.reshape((3, output.shape[2], output.shape[3]))
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.680
        output = output.transpose(1, 2, 0)
        return output
